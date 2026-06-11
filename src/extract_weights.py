import onnx
import struct
import os


def extract_weights(onnx_path, bin_path):
    model = onnx.load(onnx_path)
    weights_dict = {}
    for tensor in model.graph.initializer:
        name = tensor.name
        if tensor.raw_data:
            data = tensor.raw_data
        else:
            data = struct.pack(f'{len(tensor.float_data)}f', *tensor.float_data)

        weights_dict[name] = data

    print(f"\n[*] Zapisywanie do pliku binarnego: {bin_path}")
    with open(bin_path, 'wb') as f:
        for name, data in weights_dict.items():
            f.write(data)

    print(f"[+] Zakończono! Wagi gotowe dla C++: {bin_path}")


if __name__ == '__main__':
    yunet_onnx = "../models/face_detection_yunet_2023mar.onnx"
    yunet_bin = "../models/yunet_weights.bin"

    extract_weights(yunet_onnx, yunet_bin)