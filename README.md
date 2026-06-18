# 🎗️ PCam Metastatic Tissue Classification Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://github.com/tensorflow/tensorflow)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic-Project](https://img.shields.io/badge/UNICAM-Deep_Learning_Project-purple.svg)](https://www.unicam.it/)

This repository hosts the final project developed for the **Deep Learning** course within the Master's Degree in Computer Science (Artificial Intelligence) at the **University of Camerino (UNICAM)**. The system addresses the binary classification of metastatic histopathological patches using the **PatchCamelyon (PCam)** benchmark.

The entire workflow follows an **"incremental complexity"** paradigm: evaluating custom baseline structures, scaling up to deep residual domains (ResNet-50), exploring group-equivariant solutions (G-CNN), and ultimately leveraging out-of-domain pre-trained architectures (DenseNet121). Given the medical nature of the problem, our design explicitly **prioritizes Recall (Sensitivity)** to minimize critical False Negatives during diagnostic screening.

📌 **For a comprehensive theoretical, mathematical, and architectural breakdown, please refer to the official PDF report included in this repository:** `DL_Project.pdf`.

---

## 🔬 The Clinical Benchmark: PCam Dataset

The dataset is derived from the renowned **Camelyon16 Challenge**. Each individual patch spans $96\times96$ pixels in the H&E (Hematoxylin and Eosin) color space. Crucially, the ground-truth binary label is determined exclusively by the central $32\times32$ pixel region (the presence of at least one pixel of tumor tissue triggers a positive malignant label).

* **Experimental Subset Scale:** 80,000 balanced patches for Training, 8,000 patches for Validation (extracted maintaining a strict 50/50 class balance and fully segregated at the patient level to prevent data leakage).
* **Inference Configuration:** To evaluate the models' native diagnostic sensitivity under identical baseline constraints, all models are optimized using a symmetric loss (1:1) and cross-evaluated on the complete independent Test Set (32,768 patches) at a unified calibrated screening threshold of **0.35**.

---

## 🚀 Structural Paradigms & Pipeline Logic

The project evaluates four distinct deep learning configurations:

1. **From-Scratch CNN (Baseline):** A custom layout with progressive feature extraction blocks, highly regularized using `SpatialDropout2D` layers to prevent the network from overfitting to staining variations across slides.
2. **Deep ResNet-50:** Integration of convolutional identity shortcuts to scale network depth up to 256 internal filters while preserving deep gradient stability during backpropagation.
3. **Hybrid G-ResNet (G-CNN):** A geometric architecture that restricts and maps convolutional transformations within the $p4m$ discrete symmetry group (90° rotations and reflections). This injects a coordinate-free morphological understanding matching the isotropic nature of biological structures.
4. **Transfer Learning (DenseNet121):** Adaptation of a pre-trained ImageNet backbone to the digital pathology domain via contrast-robust data augmentation, coupled with an asymmetric cost-weighting protocol ($w_{tumor} = 2.5$) and a two-stage global partial fine-tuning strategy.

---

## 📊 Quantitative Test Set Results

All models were evaluated on the same independent test partition consisting of **32,768 patches** under the standardized $0.35$ inference threshold:

| Model Layout | Global Accuracy | Macro F1-Score | Recall (Tumor) | Precision (Tumor) | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **From-Scratch CNN** | 0.75 | 0.75 | 0.71 | 0.78 | 0.91 |
| **ResNet-50** | 0.75 | 0.75 | 0.65 | 0.82 | **0.93** |
| **Hybrid G-ResNet** | 0.83 | 0.83 | 0.77 | **0.88** | 0.92 |
| **DenseNet121 (Optimized)** | **0.84** | **0.84** | **0.85** | 0.83 | 0.91 |

### 📈 Core Academic Insights

* **The ResNet Paradox:** Despite securing the highest overall discriminative power ($\text{AUC} = 0.93$), the deep ResNet-50 initialized from scratch collapses into a rigid conservative minimum, plummeting to a critical Tumor Recall of $0.65$ as a defensive response to balance validation anomalies.
* **Geometric Regularization Utility:** By embedding $p4m$ equivariance, the G-CNN confidently handles cell morphology across all orientations, bypassing standard deep scaling limits to achieve a dominant Tumor Precision of $0.88$ using only a fraction of raw parameters.
* **The Clinical Sweet Spot (DenseNet121):** Combining specialized pre-trained feature mapping with asymmetric gradient penalties successfully unlocks the highest screening efficacy, forcing Tumor Recall up to **85%** and Accuracy to **84%**, minimizing true medical omissions without destabilizing global classification bounds.

---

## 🛠️ Infrastructure Software Engineering & I/O Optimization

To train over 300,000 global image tensors (approx. 8GB) within volatile cloud runtimes without triggering host memory failures, specific engineering constraints were implemented:
* **Out-of-Core HDF5 Streaming:** Source arrays are loaded directly from disk via binary HDF5 streams, entirely preventing host RAM duplication and virtual memory fragmentation.
* **Asynchronous Preprocessing Pipelines (`tf.data`):** Data augmentation mappings are assigned to multithreaded background host CPU workers via `AUTOTUNE`, feeding GPU registers concurrently and completely neutralizing hardware starvation bottlenecks.
* **Mixed Precision Computation (`mixed_float16`):** Tensor optimization paths are dynamically cast onto mixed 16-bit floating-point layers, maximizing the computational throughput of NVIDIA T4 Tensor Cores.
* **Just-In-Time (XLA) Compilation:** Graph compilation algorithms are turned on via `jit_compile=True` during fine-tuning to fuse internal matrix kernel nodes and accelerate optimization epochs.

---

## 👥 Project Contributors
* **Tommaso Cosci** - Master's Degree Student in Computer Science (Artificial Intelligence), UNICAM
* **Alessio Tartufoli** - Master's Degree Student in Computer Science (Artificial Intelligence), UNICAM

---
*Academic project developed for the Deep Learning course - A.Y. 2025/2026.*
