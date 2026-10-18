import onnx
from onnx import shape_inference
from collections import defaultdict

def generate_c_config(onnx_path, cfg_path):
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
                    f"{current_layer_id} {layer_type} inputs={in_str} stride={stride} pad={pad} relu={apply_relu} weights={weight_file} bias={bias_file} w_shape={w_shape_str} ref_count={ref_count}\n")

            elif op == "MaxPool":
                stride = attrs["strides"].ints[0] if "strides" in attrs else 1
                kernel = attrs["kernel_shape"].ints[0] if "kernel_shape" in attrs else 2
                pad = attrs["pads"].ints[0] if "pads" in attrs else 0
                f.write(
                    f"{current_layer_id} MAXPOOL inputs={in_str} kernel={kernel} stride={stride} pad={pad} ref_count={ref_count}\n")

            elif op in ["Resize", "Reshape"]:
                f.write(
                    f"{current_layer_id} {op.upper()} inputs={in_str} out_shape={out_shape_str} ref_count={ref_count}\n")

            elif op in ["Add", "Transpose", "Sigmoid", "Relu"]:
                f.write(f"{current_layer_id} {op.upper()} inputs={in_str} ref_count={ref_count}\n")

            tensor_to_id[out_tensor] = current_layer_id
            current_layer_id += 1


if __name__ == '__main__':
    generate_c_config("../models/face_detection_yunet_2023mar.onnx", "../models/yunet.cfg")