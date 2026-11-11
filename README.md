# Lightweight Face Recognition & Detection in C

A lightweight, high-performance facial detection and recognition pipeline written from scratch in pure C, featuring a custom graph execution engine for **YuNet** (face detection) and **SFace** (face recognition / 128D embedding extraction).

The repository provides two execution targets:
1. **Standalone CLI executable (`main.c`)** — for local debugging, accuracy verification, and benchmarks.
2. **Shared Library API (`face_engine.c` / `face_engine.h`)** — for zero-copy memory inference integrated with Python / FastAPI via `ctypes`.

> **Note:** All compiled network graphs (`.cfg`) and binary weights (`.bin`) are **already tracked in this repository**. You can immediately compile and run the C engine without downloading ONNX models or running Python compilation scripts.

---

## 🚀 Architecture Overview

1. **Inference Engine (Pure C + OpenMP)**:
   - 32-byte aligned memory management (`posix_memalign` / `_aligned_malloc`) optimized for ARM NEON and AVX cache lines.
   - Zero-copy in-place activations (`Sigmoid`, `ReLU`, `PRelu`) and O(1) `Reshape` via buffer ownership transfer.
   - Multithreaded convolutions and GEMM layers accelerated with OpenMP.
   - **Kabsch-Umeyama facial alignment**: Computes 2x3 affine transform matrices to warp and crop faces to standard 112x112 tensors.
   - **L2 Normalized Embeddings**: Produces unit-length 128D feature vectors for Cosine Similarity matching (threshold: 0.363).
2. **Shared Library & FFI Interface (`face_engine`)**:
   - Decodes images directly from in-memory JPEG/PNG buffers (`stbi_load_from_memory`) without disk I/O.
   - Clamps bounding boxes to original image dimensions in `[top, right, bottom, left]` format.
   - Exposes a clean C ABI with explicit memory lifecycle management (`free_inference_result`, `cleanup_engine`).

---

## 📁 Project Structure

```text
face_reco_for_arm/
├── CMakeLists.txt          # CMake build configuration
├── face_engine.h           # Public C API header for library consumers
├── face_engine.c           # Shared library engine implementation (memory buffer inference)
├── test_engine.py          # Python ctypes smoke test verifying shared library
├── main.c                  # Standalone CLI test & verification benchmark
├── models/
│   ├── yunet.cfg           # Compiled YuNet execution graph (tracked in git)
│   ├── yunet_weights/      # YuNet raw binary weights & biases (tracked in git)
│   ├── sface.cfg           # Compiled SFace execution graph (tracked in git)
│   └── sface_weights/      # Fused SFace raw binary weights & biases (tracked in git)
├── src/                    # Offline tools for ONNX graph compilation (optional)
│   ├── download_model.py   # Downloads official ONNX models from OpenCV Zoo
│   ├── extract_weights.py  # Dumps weights from ONNX to .bin files
│   ├── build_nn_graph.py   # Compiles execution graphs & performs operator fusion
│   └── baseline.py         # Python OpenCV reference baseline for accuracy checks
├── stb_image.h             # Single-header image decoder
├── stb_image_write.h       # Single-header image encoder (PNG/JPEG)
└── dataset/
    └── raw/                # Place test images here (e.g. roberto.jpg, roberto3.jpg)
```

---

## 🛠️ Step-by-Step Build & Execution Guide

### Prerequisite: Test Images
Image files are ignored by git. Before running tests, ensure you place reference face images inside `dataset/raw/`:
* `dataset/raw/roberto.jpg` (baseline reference image with a single face)
* `dataset/raw/roberto3.jpg` (test image to verify recognition)

---

### Path 1: Standalone CLI Executable (`main.c`)

Direct execution for local benchmarking and OpenCV accuracy comparison.

#### Linux / WSL / Raspberry Pi
```bash
# 1. Compile with OpenMP and Release optimizations
gcc -O3 -fopenmp main.c -lm -o real_version

# 2. Run the executable
./real_version
```

#### Windows (PowerShell with MinGW / GCC)
```powershell
# 1. Compile with OpenMP and Release optimizations
gcc -O3 -fopenmp main.c -lm -o real_version.exe

# 2. Run the executable
.\real_version.exe
```

#### CMake / CLion (Cross-Platform)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Linux:
./build/real_version

# Windows:
.\build\Release\real_version.exe
```

---

### Path 2: Shared Library (`face_engine`) + Python ctypes Test

Compiles the engine into a dynamic shared library (`.so` / `.dll`) and verifies memory-based inference using `test_engine.py`.

#### Linux / WSL / Raspberry Pi
```bash
# 1. Compile shared library (.so)
gcc -O3 -fopenmp -shared -fPIC face_engine.c -lm -o libfacereco.so

# 2. Setup Python environment and install numpy
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# 3. Run ctypes verification test
python3 test_engine.py
```

#### Windows (PowerShell with MinGW / GCC)
```powershell
# 1. Compile shared library (.dll)
gcc -O3 -fopenmp -shared -fPIC face_engine.c -lm -o libfacereco.dll

# 2. Setup Python environment and install numpy
python -m venv .venv
.venv\Scripts\activate
pip install numpy

# 3. Run ctypes verification test
python test_engine.py
```

#### Expected Test Output:
```text
[+] Models loaded successfully!
[+] Detected faces: 1 (error_code: 0)
    Face #1:
      - BBox [top, right, bottom, left]: [75, 302, 275, 182]
      - Confidence: 93.79%
      - Embedding: first 4 values: [-0.0589  0.0080 -0.1161  0.0282], L2 norm: 1.0000
```

---

### Path 3 (Optional): Recompiling Models from ONNX

Use this **only** if you want to re-download the official ONNX models and regenerate `.cfg` and `.bin` files from scratch.

#### 1. Setup Python Environment
```bash
# Linux / WSL:
python3 -m venv .venv
source .venv/bin/activate
pip install onnx opencv-python numpy

# Windows:
python -m venv .venv
.venv\Scripts\activate
pip install onnx opencv-python numpy
```

#### 2. Download and Recompile Graphs
```bash
cd src

# Download official ONNX models
python download_model.py

# Extract weights to binary files
python extract_weights.py

# Compile execution graphs (performs Conv+BN fusion and generates .cfg)
python build_nn_graph.py

cd ..
```

---

## 🎯 Output Heads Reference

YuNet produces predictions across 3 stride levels (strides 8, 16, and 32):

| Stride | Scale Feature Map | Cls (Face Probability) | Obj (Objectness) | BBox (dx, dy, dw, dh) | KPS (5 Landmarks) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **8**  | 80 x 80 = 6400 | `cls_8` `[1, 6400, 1]` | `obj_8` `[1, 6400, 1]` | `bbox_8` `[1, 6400, 4]` | `kps_8` `[1, 6400, 10]` |
| **16** | 40 x 40 = 1600 | `cls_16` `[1, 1600, 1]` | `obj_16` `[1, 1600, 1]` | `bbox_16` `[1, 1600, 4]` | `kps_16` `[1, 1600, 10]` |
| **32** | 20 x 20 = 400  | `cls_32` `[1, 400, 1]` | `obj_32` `[1, 400, 1]` | `bbox_32` `[1, 400, 4]` | `kps_32` `[1, 400, 10]` |

SFace processes an aligned 112x112 face crop and produces:
* Output head **`fc1`**: `[1, 128]` float32 tensor normalized with L2 norm to the unit hypersphere.
* Match criterion: Cosine Similarity >= 0.363.
