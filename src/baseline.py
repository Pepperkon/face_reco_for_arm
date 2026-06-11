import cv2
import numpy as np
import os

YUNET_MODEL = "../models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "../models/face_recognition_sface_2021dec.onnx"

THRESHOLD = 0.363


def InitModel():
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.8)
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    return detector, recognizer


def download_vector(image_path, detector, recognizer):
    'it returns 128 dimension face vector or none if there is no face...'
    if not os.path.exists(image_path):
        print("ERROR - WRONG IMAGE PATH")
        return None


    img = cv2.imread(image_path)
    if img is None:
        print("ERROR - CANNOT OPEN IMAGE")
        return None
    'resize big photo, yunet sometimes canoot handle big faces'
    MAX_SIZE = 640
    height, width, _ = img.shape
    if max(height, width) > MAX_SIZE:
        scale = MAX_SIZE / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    'set tensor size to img shape'
    height, width, _ = img.shape
    detector.setInputSize((width, height))

    status, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        print('No faces on this image :(((')
        return None

    detectedFaces = []
    for face in faces:
        aligned_face = recognizer.alignCrop(img, face)
        faceVector = recognizer.feature(aligned_face)
        detectedFaces.append(faceVector)
    return detectedFaces


def main():
    detector, recognizer = InitModel()

    photoFolder = "../dataset/raw"
    baseline_filename = "roberto.jpg"
    mainPhoto_path = os.path.join(photoFolder, baseline_filename)

    baseline_vectors = download_vector(mainPhoto_path, detector, recognizer)

    if not baseline_vectors:
        print("[-] Error - robercik does not look good today... he is unrecognizable.")
        return
    if len(baseline_vectors) > 1:
        print("[-] Error - Found more than one face in the baseline photo! Use a photo with only one person.")
        return

    roberto_vector = baseline_vectors[0]

    for filename in os.listdir(photoFolder):
        if filename == baseline_filename:
            continue
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        test_path = os.path.join(photoFolder, filename)
        test_vectors = download_vector(test_path, detector, recognizer)

        if not test_vectors:
            continue
        for i, face_vec in enumerate(test_vectors):
            score = recognizer.match(roberto_vector, face_vec, cv2.FaceRecognizerSF_FR_COSINE)

            if score >= THRESHOLD:
                print(f"    -> [Face {i + 1}] ✅ MATCH! It's Roberto (Score: {score:.4f})")
            else:
                print(f"    -> [Face {i + 1}] ❌ INTRUDER (Score: {score:.4f})")


if __name__ == "__main__":
    main()
