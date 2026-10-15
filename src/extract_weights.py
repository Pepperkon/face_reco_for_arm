import onnx
import struct
import os


def extract_weights(onnx_path, out_dir):
    model = onnx.load(onnx_path)
    os.makedirs(out_dir, exist_ok=True)

    shapes_file = open(os.path.join(out_dir, "shapes.txt"), "w")

    for tensor in model.graph.initializer:
        safe_name = tensor.name.replace("/", "_").replace(":", "_")
        bin_path = os.path.join(out_dir, f"{safe_name}.bin")

        if tensor.raw_data:
            data = tensor.raw_data
        else:
            data = struct.pack(f'{len(tensor.float_data)}f', *tensor.float_data)

        with open(bin_path, 'wb') as f:
            f.write(data)

        dims_str = "x".join(map(str, tensor.dims))
        shapes_file.write(f"{safe_name}.bin : [{dims_str}]\n")
        print(f"Saved: {safe_name}.bin [{dims_str}]")

    shapes_file.close()


if __name__ == '__main__':
    yunet_onnx = "../models/face_detection_yunet_2023mar.onnx"
    out_dir = "../models/yunet_weights"

    extract_weights(yunet_onnx, out_dir)