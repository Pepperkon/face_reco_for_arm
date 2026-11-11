import ctypes
import os

import numpy as np


class CDetectedFace(ctypes.Structure):
    _fields_ = [
        ("top", ctypes.c_int),
        ("right", ctypes.c_int),
        ("bottom", ctypes.c_int),
        ("left", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("embedding", ctypes.c_float * 128),
    ]


class CInferenceResult(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("error_code", ctypes.c_int),
        ("faces", ctypes.POINTER(CDetectedFace)),
    ]


# Select library (.dll on Windows, .so on Linux)
lib_name = "./libfacereco.dll" if os.name == "nt" else "./libfacereco.so"
lib = ctypes.CDLL(lib_name)

lib.init_engine.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
]
lib.init_engine.restype = ctypes.c_int

lib.process_image_memory.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
]
lib.process_image_memory.restype = CInferenceResult

lib.free_inference_result.argtypes = [ctypes.POINTER(CInferenceResult)]
lib.free_inference_result.restype = None

# Initialize models
ret = lib.init_engine(
    b"models/yunet.cfg",
    b"models/yunet_weights",
    b"models/sface.cfg",
    b"models/sface_weights",
    4,
)
assert ret == 0, "Failed to initialize models!"
print("[+] Models loaded successfully!")

# Load image bytes into RAM
with open("dataset/raw/roberto.jpg", "rb") as f:
    img_bytes = f.read()

c_buf = (ctypes.c_ubyte * len(img_bytes)).from_buffer_copy(img_bytes)
res = lib.process_image_memory(c_buf, len(img_bytes), 0.6, 0.3)

try:
    print(f"[+] Detected faces: {res.count} (error_code: {res.error_code})")
    for i in range(res.count):
        face = res.faces[i]
        emb = np.ctypeslib.as_array(face.embedding)
        norm = np.linalg.norm(emb)
        print(f"    Face #{i + 1}:")
        print(
            f"      - BBox [top, right, bottom, left]: [{face.top}, {face.right}, {face.bottom}, {face.left}]"
        )
        print(f"      - Confidence: {face.confidence:.2%}")
        print(f"      - Embedding: first 4 values: {emb[:4]}, L2 norm: {norm:.4f}")
finally:
    lib.free_inference_result(ctypes.byref(res))
