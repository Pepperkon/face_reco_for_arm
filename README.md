# Lightweight Face Recognition & Detection in C

A lightweight, high-performance facial detection and recognition pipeline written from scratch in pure C, featuring a custom graph execution engine for **YuNet** (face detection) and **SFace** (face recognition).

---

## 🚀 Architecture Overview

1. **Model Preparation (Python)**:
   - Downloads official ONNX models (`YuNet` and `SFace`).
   - Extracts binary float32 tensor weights into flat binary files (`.bin`).
   - Parses the ONNX computational graph, performs Conv+ReLU operator fusion, and generates a compact linear execution configuration (`yunet.cfg`).
2. **Inference Engine (Pure C)**:
   - 32-byte aligned memory management (`posix_memalign` / `_aligned_malloc`) optimized for ARM NEON and AVX cache lines.
   - Zero-copy / in-place activations (`Sigmoid`, `ReLU`) and $O(1)$ `Reshape` via buffer ownership transfer (no redundant memory allocations).
   - Generates multi-scale predictions matching official OpenCV DNN float precision.

---

## 📁 Project Structure

```text
lightweight-face-recognition/
├── CMakeLists.txt          # CMake build configuration
├── main.c                  # Core neural network inference engine in C
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   ├── yunet.cfg           # Generated network graph configuration
│   └── yunet_weights/      # Raw binary weights & biases (.bin)
├── src/
│   ├── download_model.py   # Downloads ONNX models from OpenCV Zoo
│   ├── extract_weights.py  # Dumps weights from ONNX to .bin files
│   ├── build_nn_graph.py   # Graph compiler: operator fusion & .cfg generation
│   └── baseline.py         # Python OpenCV reference baseline
└── dataset/                # Test images
```

---

## 🛠️ Step-by-Step Setup & Execution

### 1. Python Environment Setup

Make sure you have Python 3.9+ with `onnx` and `opencv-python` installed:

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install onnx opencv-python numpy
```

---

### 2. Download ONNX Models

Run the download script to fetch the latest official YuNet and SFace models:

```powershell
cd src
python download_model.py
```

*This downloads the models directly into `models/`.*

---

### 3. Extract Weights to Binary Files

Extract all network weight and bias tensors into raw float32 binary `.bin` files:

```powershell
python extract_weights.py
```

*This saves all layer weights into `models/yunet_weights/`.*

---

### 4. Compile the Network Graph (`.cfg`)

Compile the ONNX computation graph into the lightweight engine configuration:

```powershell
python build_nn_graph.py
cd ..
```

*This generates `models/yunet.cfg` with fused Conv+ReLU layers, memory lifetime counts (`ref_count`), and output head tags.*

---

### 5. Build and Run the C Engine

#### Option A: Direct GCC Compilation (Recommended for testing)

```powershell
# Compile with Release optimizations (-O3)
gcc -O3 -Wall -fopenmp main.c -lm -o real_version

# Run the inference engine
.\real_version.exe
```

#### Option B: CMake / CLion

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\real_version.exe
```

> [!NOTE]
> When running from inside CLion, ensure the **Working Directory** in *Run/Debug Configurations* is set to the project root directory (`$PROJECT_DIR$`), or rely on the automatic `../` fallback built into `load_network()`.

---

## 📊 Expected Output

The engine loads the 91-layer graph, loads all binary weights into memory, and performs a full forward pass on a 640x640 tensor:

```text
[*] Loading network...
[+] Network loaded successfully! Num layers: 91
[*] Creating dummy input [1, 3, 640, 640]...
[*] Running forward pass...
[+] Forward pass completed! Output count: 12
    -> Output [0]: 'cls_8' shape: [1, 1, 6400, 1], first val: 0.6219
    -> Output [1]: 'cls_16' shape: [1, 1, 1600, 1], first val: 0.5432
    -> Output [2]: 'cls_32' shape: [1, 1, 400, 1], first val: 0.5871
    -> Output [3]: 'obj_8' shape: [1, 1, 6400, 1], first val: 0.0000
    -> Output [4]: 'obj_16' shape: [1, 1, 1600, 1], first val: 0.0000
    -> Output [5]: 'obj_32' shape: [1, 1, 400, 1], first val: 0.0000
    -> Output [6]: 'bbox_8' shape: [1, 1, 6400, 4], first val: 0.3640
    -> Output [7]: 'bbox_16' shape: [1, 1, 1600, 4], first val: 0.5072
    -> Output [8]: 'bbox_32' shape: [1, 1, 400, 4], first val: 1.3809
    -> Output [9]: 'kps_8' shape: [1, 1, 6400, 10], first val: -0.1562
    -> Output [10]: 'kps_16' shape: [1, 1, 1600, 10], first val: 0.2290
    -> Output [11]: 'kps_32' shape: [1, 1, 400, 10], first val: -0.1462
[+] Cleaned up memory. All good!
```

---

## 🎯 Output Heads Reference

YuNet produces predictions across 3 stride levels (strides 8, 16, and 32):

| Stride | Scale Feature Map | Cls (Face Probability) | Obj (Objectness) | BBox ($dx, dy, dw, dh$) | KPS (5 Landmarks) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **8**  | $80 \times 80 = 6400$ | `cls_8` `[1, 6400, 1]` | `obj_8` `[1, 6400, 1]` | `bbox_8` `[1, 6400, 4]` | `kps_8` `[1, 6400, 10]` |
| **16** | $40 \times 40 = 1600$ | `cls_16` `[1, 1600, 1]` | `obj_16` `[1, 1600, 1]` | `bbox_16` `[1, 1600, 4]` | `kps_16` `[1, 1600, 10]` |
| **32** | $20 \times 20 = 400$  | `cls_32` `[1, 400, 1]` | `obj_32` `[1, 400, 1]` | `bbox_32` `[1, 400, 4]` | `kps_32` `[1, 400, 10]` |
