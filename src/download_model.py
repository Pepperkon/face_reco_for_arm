import os
import urllib.request


def download_model(url, save_path, model_name):
    if os.path.exists(save_path):
        print(f"[*] {model_name} IT'S ALREADY ON DISK {save_path}")
        return

    print(f"[*] DOWNLOADING {model_name}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        print(f"SUCCES {save_path}")
    except Exception as e:
        print(f"ERROR {model_name}: {e}")


def main():
    url_yunet = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    url_sface = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

    save_path_yunet = "../models/face_detection_yunet_2023mar.onnx"
    save_path_sface = "../models/face_recognition_sface_2021dec.onnx"

    print("--- DOWNLOAD MODELS ---")

    download_model(url_yunet, save_path_yunet, "YuNet (face detection)")
    download_model(url_sface, save_path_sface, "SFace (face recognition)")

    print("--- MODELS ON THE DISK ---")


if __name__ == "__main__":
    main()