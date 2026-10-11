import onnx

def generate_c_config(onnx_path, cfg_path):
    model = onnx.load(onnx_path)
    tensor_to_id = {model.graph.input[0].name: -1}
    current_layer_id = 0

    with open(cfg_path, 'w') as f:
        for node in model.graph.node:
            op = node.op_type

            inputs = []
            for inp in node.input:
                if inp in tensor_to_id:
                    inputs.append(str(tensor_to_id[inp]))

            in_str = ",".join(inputs) if inputs else "-1"

            if op == "Conv":
                group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
                if group > 1:
                    f.write(f"{current_layer_id} DEPTHWISE inputs={in_str} name={node.name}\n")
                else:
                    f.write(f"{current_layer_id} CONV inputs={in_str} name={node.name}\n")

            elif op == "MaxPool":
                f.write(f"{current_layer_id} MAXPOOL inputs={in_str}\n")
            elif op == "Add":
                f.write(f"{current_layer_id} ADD inputs={in_str}\n")
            elif op == "Resize":
                f.write(f"{current_layer_id} RESIZE inputs={in_str}\n")
            elif op == "Transpose":
                f.write(f"{current_layer_id} TRANSPOSE inputs={in_str}\n")
            elif op == "Reshape":
                f.write(f"{current_layer_id} RESHAPE inputs={in_str}\n")
            elif op == "Sigmoid":
                f.write(f"{current_layer_id} SIGMOID inputs={in_str}\n")
            elif op == "Relu":
                f.write(f"{current_layer_id} RELU inputs={in_str}\n")

            tensor_to_id[node.output[0]] = current_layer_id
            current_layer_id += 1

if __name__ == '__main__':
    generate_c_config("../models/face_detection_yunet_2023mar.onnx", "../models/yunet.cfg")