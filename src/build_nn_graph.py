import os
import onnx
from onnx import shape_inference, numpy_helper
import numpy as np
from collections import defaultdict


def save_tensor_bin(array: np.ndarray, path: str, shapes_log=None):
    """Zapisuje tensor w formacie binarnym float32 oraz opcjonalnie odnotowuje wymiary."""
    arr = array.astype(np.float32)
    arr.tofile(path)
    if shapes_log is not None:
        filename = os.path.basename(path)
        dims_str = "x".join(map(str, arr.shape))
        shapes_log.write(f"{filename} : [{dims_str}]\n")

def generate_config_yunet(onnx_path, cfg_path):
    model = onnx.load(onnx_path)
    # import this because I am to lazy to calculate this in engine
    model = shape_inference.infer_shapes(model)

    # change neurons in the graph, so when relu is after conv layer, it is combined in one big neuron
    # we track how many times each neuron is used because we cannot fuse conv with relu if conv is used more times
    initial_uses = defaultdict(int)
    producer = {}
    for node in model.graph.node:
        producer[node.output[0]] = node
        for i in node.input:
            initial_uses[i]+=1

    #track fused relus outputs and convs that are fused
    fused_relu_outs={}
    fused_conv = set()

    for node in model.graph.node:
        if node.op_type == "Relu":
            inp = node.input[0]
            prod = producer.get(inp) # now we have prod -> inp
            #fusion only when conv->relu end relu is used one time as an input
            if prod and prod.op_type == "Conv" and  initial_uses[inp]==1:
                fused_relu_outs[node.output[0]] = inp
                fused_conv.add(inp)

    #count usage after fusions
    usage_count = defaultdict(int)
    for node in model.graph.node:
        #check if it is not a relu that was fused
        if node.output[0] in fused_relu_outs:
            continue
        for i in node.input:
            actual_inp = i
            #later we will add batchnorm that is combined to that is why while and no if or smth like this
            #an this is for fusing when relu disapeers
            while actual_inp in fused_relu_outs:
                actual_inp = fused_relu_outs[actual_inp]
            usage_count[actual_inp] +=1

    all_shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        all_shapes[vi.name] = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    tensor_shapes = {init.name: init.dims for init in model.graph.initializer}

    tensor_to_id = {model.graph.input[0].name: -1}
    current_layer_id = 0

    graph_output_names = {out.name for out in model.graph.output}
    with open(cfg_path, 'w') as f:
        for node in model.graph.node:
            out_tensor = node.output[0]

            #if fused relu we skip it
            if out_tensor in fused_relu_outs:
                #conv out is fused conv neuron
                conv_out = fused_relu_outs[out_tensor]
                # we give out tensor
                tensor_to_id[out_tensor] = tensor_to_id[conv_out]
                continue

            op = node.op_type
            ref_count = usage_count.get(out_tensor, 0)
            out_shape = all_shapes.get(out_tensor, [1, 1, 1, 1])
            out_shape_str = ",".join(map(str, out_shape))

            inputs = []
            for inp in node.input:
                actual_inp = inp
                while actual_inp in fused_relu_outs:
                    actual_inp = fused_relu_outs[actual_inp]
                if actual_inp in tensor_to_id:
                    inputs.append(str(tensor_to_id[actual_inp]))
            in_str = ",".join(inputs) if inputs else "-1"

            attrs = {attr.name: attr for attr in node.attribute}
            out_tag = f" out_name={out_tensor}" if out_tensor in graph_output_names else ""
            if op == "Conv":
                group = attrs["group"].i if "group" in attrs else 1
                stride = attrs["strides"].ints[0] if "strides" in attrs else 1
                pad = attrs["pads"].ints[0] if "pads" in attrs else 0

                apply_relu = 1 if out_tensor in fused_conv else 0

                weight_file = node.input[1].replace("/", "_").replace(":", "_") + ".bin"
                bias_file = node.input[2].replace("/", "_").replace(":", "_") + ".bin" if len(
                    node.input) > 2 else "NULL"
                w_shape = tensor_shapes.get(node.input[1], [1, 1, 1, 1])
                w_shape_str = ",".join(map(str, w_shape))

                layer_type = "DEPTHWISE" if group > 1 else "CONV"

                f.write(
                    f"{current_layer_id} {layer_type} inputs={in_str} stride={stride} pad={pad} relu={apply_relu} weights={weight_file} bias={bias_file} w_shape={w_shape_str} ref_count={ref_count}{out_tag}\n")

            elif op == "MaxPool":
                stride = attrs["strides"].ints[0] if "strides" in attrs else 1
                kernel = attrs["kernel_shape"].ints[0] if "kernel_shape" in attrs else 2
                pad = attrs["pads"].ints[0] if "pads" in attrs else 0
                f.write(
                    f"{current_layer_id} MAXPOOL inputs={in_str} kernel={kernel} stride={stride} pad={pad} ref_count={ref_count}{out_tag}\n")

            elif op in ["Resize", "Reshape"]:
                f.write(
                    f"{current_layer_id} {op.upper()} inputs={in_str} out_shape={out_shape_str} ref_count={ref_count}{out_tag}\n")

            elif op in ["Add", "Transpose", "Sigmoid", "Relu"]:
                f.write(f"{current_layer_id} {op.upper()} inputs={in_str} ref_count={ref_count}{out_tag}\n")

            tensor_to_id[out_tensor] = current_layer_id
            current_layer_id += 1




def generate_sface_config(onnx_path, cfg_path, weights_dir):
    print(f"[*] Przetwarzanie SFace: {onnx_path} -> {cfg_path}")
    print(f"[*] Katalog wag wyjściowych: {weights_dir}")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(cfg_path)), exist_ok=True)

    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)

    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    shapes_file = open(os.path.join(weights_dir, "shapes_fused.txt"), "w")

    nodes = list(model.graph.node)
    producer = {n.output[0]: n for n in nodes}

    conv_to_bn = {}
    for i, n in enumerate(nodes):
        if n.op_type == "Conv":
            next_node = nodes[i + 1] if i + 1 < len(nodes) else None
            if next_node and next_node.op_type == "BatchNormalization":
                conv_to_bn[n.output[0]] = next_node

    tensor_to_id = {model.graph.input[0].name: -1}
    current_layer_id = 0

    cfg_lines = []

    sub_node = nodes[0]
    mul_node = nodes[1]
    assert sub_node.op_type == "Sub" and mul_node.op_type == "Mul"

    val_sub = float(inits[sub_node.input[1]][0])
    val_mul = float(inits[mul_node.input[1]][0])

    cfg_lines.append(f"{current_layer_id} SUB_SCALAR inputs=-1 scalar={val_sub} ref_count=1")
    tensor_to_id[sub_node.output[0]] = current_layer_id
    current_layer_id += 1

    cfg_lines.append(f"{current_layer_id} MUL_SCALAR inputs={current_layer_id - 1} scalar={val_mul} ref_count=1")
    tensor_to_id[mul_node.output[0]] = current_layer_id
    current_layer_id += 1

    i = 2
    while i < len(nodes):
        node = nodes[i]

        if node.op_type == "BatchNormalization" and node.name == "bn1":
            break

        if node.op_type == "Conv":
            bn_node = nodes[i + 1]
            assert bn_node.op_type == "BatchNormalization"
            prelu_node = nodes[i + 2]
            assert prelu_node.op_type == "PRelu"

            attrs_conv = {a.name: a for a in node.attribute}
            group = attrs_conv["group"].i if "group" in attrs_conv else 1
            stride = attrs_conv["strides"].ints[0] if "strides" in attrs_conv else 1
            pad = attrs_conv["pads"].ints[0] if "pads" in attrs_conv else 0
            layer_type = "DEPTHWISE" if group > 1 else "CONV"

            w_raw = inits[node.input[1]].copy()
            gamma = inits[bn_node.input[1]]
            beta = inits[bn_node.input[2]]
            mean = inits[bn_node.input[3]]
            var = inits[bn_node.input[4]]

            attrs_bn = {a.name: a for a in bn_node.attribute}
            eps = attrs_bn["epsilon"].f if "epsilon" in attrs_bn else 1e-3

            scale = gamma / np.sqrt(var + eps)
            w_fused = w_raw * scale[:, None, None, None]
            b_fused = beta - mean * scale

            prefix = node.name.replace("/", "_").replace(":", "_")
            w_file = f"{prefix}_fused_weight.bin"
            b_file = f"{prefix}_fused_bias.bin"
            save_tensor_bin(w_fused, os.path.join(weights_dir, w_file), shapes_file)
            save_tensor_bin(b_fused, os.path.join(weights_dir, b_file), shapes_file)

            w_shape_str = ",".join(map(str, w_fused.shape))

            in_layer_id = tensor_to_id[node.input[0]]
            cfg_lines.append(
                f"{current_layer_id} {layer_type} inputs={in_layer_id} stride={stride} pad={pad} "
                f"relu=0 weights={w_file} bias={b_file} w_shape={w_shape_str} ref_count=1"
            )
            conv_layer_id = current_layer_id
            current_layer_id += 1

            slope_tensor_name = prelu_node.input[1]
            slope_file = slope_tensor_name.replace("/", "_").replace(":", "_") + ".bin"
            slope_data = inits[slope_tensor_name].squeeze()
            save_tensor_bin(slope_data, os.path.join(weights_dir, slope_file), shapes_file)

            slope_len = len(slope_data)
            cfg_lines.append(
                f"{current_layer_id} PRELU inputs={conv_layer_id} bias={slope_file} slope_len={slope_len} ref_count=1"
            )
            tensor_to_id[prelu_node.output[0]] = current_layer_id
            current_layer_id += 1

            i += 3
            continue

        i += 1

    bn1_node = nodes[i]  # węzeł 83
    dropout_node = nodes[i + 1]  # węzeł 84
    flatten_node = nodes[i + 2]  # węzeł 85
    gemm_node = nodes[i + 3]  # węzeł 86
    fc1_node = nodes[i + 4]  # węzeł 87

    assert bn1_node.op_type == "BatchNormalization"
    assert dropout_node.op_type == "Dropout"
    assert flatten_node.op_type == "Flatten"
    assert gemm_node.op_type == "Gemm"
    assert fc1_node.op_type == "BatchNormalization"

    last_prelu_id = tensor_to_id[bn1_node.input[0]]
    cfg_lines.append(f"{current_layer_id} FLATTEN inputs={last_prelu_id} ref_count=1")
    flatten_layer_id = current_layer_id
    current_layer_id += 1

    W_orig = inits[gemm_node.input[1]].copy()  # [128, 50176]
    b_orig = inits[gemm_node.input[2]].copy()  # [128]

    gamma_fc1 = inits[fc1_node.input[1]]
    beta_fc1 = inits[fc1_node.input[2]]
    mean_fc1 = inits[fc1_node.input[3]]
    var_fc1 = inits[fc1_node.input[4]]
    eps_fc1 = 2e-5

    scale_fc1 = gamma_fc1 / np.sqrt(var_fc1 + eps_fc1)
    W_fused = W_orig * scale_fc1[:, None]
    b_fused = beta_fc1 + (b_orig - mean_fc1) * scale_fc1

    gamma_bn1 = inits[bn1_node.input[1]]
    beta_bn1 = inits[bn1_node.input[2]]
    mean_bn1 = inits[bn1_node.input[3]]
    var_bn1 = inits[bn1_node.input[4]]
    eps_bn1 = 2e-5

    scale_bn1 = gamma_bn1 / np.sqrt(var_bn1 + eps_bn1)
    shift_bn1 = beta_bn1 - mean_bn1 * scale_bn1

    scale_bn1_expanded = np.repeat(scale_bn1, 49)
    shift_bn1_expanded = np.repeat(shift_bn1, 49)

    b_final = b_fused + np.dot(W_fused, shift_bn1_expanded)
    W_final = W_fused * scale_bn1_expanded[None, :]

    w_gemm_file = "pre_fc1_fused_weight.bin"
    b_gemm_file = "pre_fc1_fused_bias.bin"
    save_tensor_bin(W_final, os.path.join(weights_dir, w_gemm_file), shapes_file)
    save_tensor_bin(b_final, os.path.join(weights_dir, b_gemm_file), shapes_file)

    w_gemm_shape_str = f"{W_final.shape[0]},{W_final.shape[1]},1,1"

    cfg_lines.append(
        f"{current_layer_id} GEMM inputs={flatten_layer_id} weights={w_gemm_file} bias={b_gemm_file} "
        f"w_shape={w_gemm_shape_str} ref_count=0 out_name=fc1"
    )

    shapes_file.close()

    # Zapis do pliku konfiguracyjnego
    with open(cfg_path, "w") as f:
        for line in cfg_lines:
            f.write(line + "\n")

    print(f"[+] Zapisano SFace cfg: {cfg_path} ({len(cfg_lines)} warstw, 0 warstw BatchNorm)")

def verify_sface_fusion(onnx_path, cfg_path, weights_dir):
    try:
        import cv2
    except ImportError:
        return

    net = cv2.dnn.readNet(onnx_path)

    np.random.seed(42)
    dummy_input = np.random.uniform(0, 255, (1, 3, 112, 112)).astype(np.float32)

    net.setInput(dummy_input)
    ref_out = net.forward()

    x = dummy_input.copy()
    # SUB + MUL
    x = (x - 127.5) * 0.0078125

    with open(cfg_path, "r") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 3]

    for line in lines:
        parts = line.split()
        l_type = parts[1]

        if l_type in ["CONV", "DEPTHWISE"]:
            # Parsowanie parametrów
            w_file = [p.split("=")[1] for p in parts if p.startswith("weights=")][0]
            b_file = [p.split("=")[1] for p in parts if p.startswith("bias=")][0]
            w_shape = list(map(int, [p.split("=")[1] for p in parts if p.startswith("w_shape=")][0].split(",")))
            stride = int([p.split("=")[1] for p in parts if p.startswith("stride=")][0])
            pad = int([p.split("=")[1] for p in parts if p.startswith("pad=")][0])

            w_data = np.fromfile(os.path.join(weights_dir, w_file), dtype=np.float32).reshape(w_shape)
            b_data = np.fromfile(os.path.join(weights_dir, b_file), dtype=np.float32)

            # Prosta konwolucja w cv2/numpy
            c_out, c_in, kh, kw = w_shape
            _, _, h, w_in = x.shape
            out_h = (h - kh + 2 * pad) // stride + 1
            out_w = (w_in - kw + 2 * pad) // stride + 1

            if pad > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
            else:
                x_padded = x

            out = np.zeros((1, c_out, out_h, out_w), dtype=np.float32)
            if l_type == "DEPTHWISE":
                for c in range(c_out):
                    for oy in range(out_h):
                        for ox in range(out_w):
                            patch = x_padded[0, c, oy * stride:oy * stride + kh, ox * stride:ox * stride + kw]
                            out[0, c, oy, ox] = np.sum(patch * w_data[c, 0]) + b_data[c]
            else:
                for oc in range(c_out):
                    for oy in range(out_h):
                        for ox in range(out_w):
                            patch = x_padded[0, :, oy * stride:oy * stride + kh, ox * stride:ox * stride + kw]
                            out[0, oc, oy, ox] = np.sum(patch * w_data[oc]) + b_data[oc]
            x = out

        elif l_type == "PRELU":
            slope_file = [p.split("=")[1] for p in parts if p.startswith("bias=")][0]
            slope = np.fromfile(os.path.join(weights_dir, slope_file), dtype=np.float32).reshape(1, -1, 1, 1)
            x = np.where(x > 0, x, x * slope)

        elif l_type == "FLATTEN":
            x = x.reshape(1, -1)

        elif l_type == "GEMM":
            w_file = [p.split("=")[1] for p in parts if p.startswith("weights=")][0]
            b_file = [p.split("=")[1] for p in parts if p.startswith("bias=")][0]
            w_shape = list(map(int, [p.split("=")[1] for p in parts if p.startswith("w_shape=")][0].split(",")))
            w_data = np.fromfile(os.path.join(weights_dir, w_file), dtype=np.float32).reshape(w_shape[0], w_shape[1])
            b_data = np.fromfile(os.path.join(weights_dir, b_file), dtype=np.float32)
            x = np.dot(x, w_data.T) + b_data

    max_diff = np.max(np.abs(ref_out - x))
    print(f"[+] Weryfikacja zakończona sukcesem! Maksymalna różnica bezwzględna względem cv2.dnn: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Zbyt duża rozbieżność: {max_diff}"


if __name__ == '__main__':
    yunet_onnx = "models/face_detection_yunet_2023mar.onnx"
    yunet_cfg = "models/yunet.cfg"

    sface_onnx = "models/face_recognition_sface_2021dec.onnx"
    sface_cfg = "models/sface.cfg"
    sface_weights_dir = "models/sface_weights"

    # Sprawdzenie ścieżek relatywnych (gdy skrypt uruchamiany jest z katalogu głównego lub src/)
    if not os.path.exists(yunet_onnx):
        yunet_onnx = "../models/face_detection_yunet_2023mar.onnx"
        yunet_cfg = "../models/yunet.cfg"
        sface_onnx = "../models/face_recognition_sface_2021dec.onnx"
        sface_cfg = "../models/sface.cfg"
        sface_weights_dir = "../models/sface_weights"

    if os.path.exists(yunet_onnx):
        generate_config_yunet(yunet_onnx, yunet_cfg)

    if os.path.exists(sface_onnx):
        generate_sface_config(sface_onnx, sface_cfg, sface_weights_dir)
        verify_sface_fusion(sface_onnx, sface_cfg, sface_weights_dir)