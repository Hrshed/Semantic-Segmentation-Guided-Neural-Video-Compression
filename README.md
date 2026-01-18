# Semantic Segmentation Guided Neural Video Compression 

https://github.com/user-attachments/assets/8c6f7c55-0cbe-4799-9fb3-d7b2b816d70d

This repository contains the code for the **“Semantic Segmentation Guided Neural Video Compression”**.

The project extends a neural video compression baseline with semantic conditioning and a “Performance” architecture optimized for foreground regions. The final model improves **ROI PSNR by ~1–1.5 dB over the baseline at similar BPP**, with approximately **14% runtime overhead** (see accompanying report for details).

This README documents the environment, training and evaluation procedure so that the main results of the report can be reproduced.

---

## 1. Repository structure

The most relevant files for reproduction are:

- `docker-compose.yaml` – Docker service definition for training & Jupyter for graphing/testing the models 
- `Dockerfile` – Build recipe for the runtime environment (Python, CUDA, `uv`, etc.).
- `requirements.txt` – Python dependencies used inside the container
- `video_compression_config.yaml` – Configuration file for dataset paths, model and training hyperparameters.
- `trainer_seg_video_model.py` – Main training script for all models including baseline.
- `report_graphs.ipynb` – Jupyter notebook with the plotting code used to generate the comparison figures in the report.
- `checkpoints/` – Directory for original model weights (mounted into the container).
- `logs/` – Training logs, metrics and our checkpoints.

> **Note:** Exact internals of the model (modules, datasets, etc.) are described in the report; this README focuses on how to run the provided implementation.

---

## 2. Environment setup

### 2.1. Recommended: Docker + NVIDIA (reproducible)

The easiest and most reproducible way to run the code is via the provided `docker-compose.yaml`.

#### Prerequisites

- Docker
- NVIDIA GPU and NVIDIA Container Toolkit (for `runtime: nvidia`)
- Access to the dataset as described in the report (mounted into the container)

#### Environment variables

The `docker-compose.yaml` expects the following environment variables on the host:

- `DATA_PATH` – path on the host where the official Waymo **training split** is located.
- `TEST_DATA_PATH` – path on the host where the official Waymo **test split** is located.
- `PROJECT_PATH` – path on the host where you want to store checkpoints and logs.

Example (Linux / macOS):

```bash
export DATA_PATH=/path/to/your/train_split
export TEST_DATA_PATH=/path/to/your/test_split
export PROJECT_PATH=/path/to/this/repo

mkdir -p "$PROJECT_PATH/checkpoints" "$PROJECT_PATH/logs"
```

### 2.2. Build and start the container

```bash
docker compose up
```
This will build the image and start the main container

## 3. Training models

Training is done via the trainer_seg_video_model.py script, using uv as the Python runner.

In the code or comments you may see the training command written with a leading # (commented).
To actually run training, remove the leading # and execute:
```video_compression_config.yaml
uv run /workspace/trainer_seg_video_model.py num_gpus=1
```
3.1. Selecting which model variant to train

The P-frame (video) model variant is controlled by the `dmc_variant` field in `video_compression_config.yaml`.

Valid values:

"old" – baseline DMC.

"performance" – Performance architecture (foreground-optimized).

"fast" – speed-optimized variant.

"mask_prop" – mask-propagation variant (enables mask propagation logic; internally sets MASK_PROP=True).

## 4. Generating results & plots via Jupyter
The report figures and qualitative examples are generated using the report_graphs.ipynb notebook inside the container.

In the documentation or compose file you may see the Jupyter command commented out.
To start Jupyter Lab inside the container, remove the leading # and run:

```
command: uv run --with jupyter jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
```

This will start a Jupyter Lab server listening on port 8888 inside the container.

On the host machine, open your browser and navigate to:

http://localhost:8888/


You should now see Jupyter Lab and be able to open report_graphs.ipynb to:

Load trained checkpoints from logs/.

Re-generate the comparison plots used in the report.

Inspect rate–distortion curves, ROI PSNR, qualitative reconstructions, etc.
