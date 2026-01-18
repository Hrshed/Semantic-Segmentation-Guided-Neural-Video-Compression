# Converting Docker Container for LRZ Enroot

This guide walks you through converting your Docker container to work with the LRZ AI cluster's Enroot system.

## Prerequisites

- Docker access on your local machine or build server
- LRZ AI cluster access with your username
- Your Dockerfile ready for building

## Step 1: Build Docker Image Locally

On your local machine (or wherever you have Docker access):

```bash
# Set your project variables
PROJECT_NAME="multimodalneuralvideocompression"
VERSION="latest"
DOCKER_IMAGE_NAME="${PROJECT_NAME}:${VERSION}"

# Build the Docker image
docker build -t ${DOCKER_IMAGE_NAME} .
```

## Step 2: Choose Transfer Method

You have two options for getting your image to LRZ:

### Option A: Private Registry

```bash
# Login to LRZ GitLab registry
docker login gitlab.lrz.de:5005

# Tag and push image
docker tag ${DOCKER_IMAGE_NAME} gitlab.lrz.de:5005/gitlab.lrz.de:5005/teleoperiertes_fahren/research_krauss/multimodalneuralvideocompression/${DOCKER_IMAGE_NAME}
docker push gitlab.lrz.de:5005/gitlab.lrz.de:5005/teleoperiertes_fahren/research_krauss/multimodalneuralvideocompression/${DOCKER_IMAGE_NAME}

# Logout
docker logout gitlab.lrz.de:5005
```


## Step 3: Convert to Enroot Format

SSH into the LRZ AI cluster:

```bash
ssh go29sen2@login.ai.lrz.de
```

Start an interactive session:

```bash
salloc -p lrz-hgx-h100-94x1 --gres=gpu:1 --time=01:00:00
srun --pty bash
```

### If using Registry (Option A):

```bash
# Import from registry
enroot import --output multimodalneuralvideocompression+latest.sqsh docker://gitlab.lrz.de:5005/teleoperiertes_fahren/research_krauss/multimodalneuralvideocompression/multimodalneuralvideocompression:latest
```

## Step 4: Move to DSS Storage

Move the converted image to your DSS home directory:

```bash
# Replace 'go29sen2' with your actual LRZ username
mv multimodalneuralvideocompression+latest.sqsh /dss/dsshome1/lxc0E/go29sen2/
```

## Step 5: Test the Container (Optional)

Test that your container works:

```bash
enroot start --root /dss/dsshome1/lxc0E/go29sen2/multimodalneuralvideocompression+latest.sqsh bash

# Inside the container, test a simple command
python --version
uv --version

# Exit the container
exit
```

Exit the interactive session:

```bash
exit  # Exit srun
exit  # Exit salloc
```

## Step 6: Update Your SLURM Script

Create your updated SLURM batch script:

```bash
#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=video-compression-train
#SBATCH --time=2-00:00:00
#SBATCH -o out/video-compression-%j.out
#SBATCH -e out/video-compression-%j.err

# Set environment variables
export OMP_NUM_THREADS=96
export MKL_NUM_THREADS=8

srun --mpi=pmi2 \
    --container-mounts=$DSS_PROJECT_PATH:/workspace,$DSS_DATA_PATH:/workspace/dataset \
    --container-image=/dss/dsshome1/08/go29sen2/xuan/multimodalneuralvideocompression/multimodalneuralvideocompression+latest.sqsh \
    python /workspace/trainer_video_model.py

echo "Done training!"
```

## Step 7: Set Environment Variables and Submit

Set your DSS paths (adjust these to your actual paths):

```bash
export DSS_PROJECT_PATH="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/go29sen2/"
export DSS_DATA_PATH="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/waymo_v_1_4_2"
```

Submit your job:

```bash
sbatch your_job_script.sbatch
```

## Important Notes

- **Replace `go29sen2`** with your actual LRZ username in all paths
- **Update DSS paths** to match your actual project and data locations
- **The `.sqsh` extension** is required for Enroot images
- **Container mounts** map external DSS paths to internal container paths
- **Use `--mpi=pmi2`** if your application uses multiprocessing (like PyTorch DataLoaders)

## Troubleshooting

### Container Import Fails
- Check if you have enough space in your DSS home directory
- Verify the image name and tag are correct
- Try importing a smaller test image first

### Container Won't Start
- Test with a simple command like `bash` instead of your application
- Check that all mount paths exist on the DSS filesystem
- Verify CUDA/GPU access with `nvidia-smi` inside the container

### Job Fails to Submit
- Check partition availability with `sinfo`
- Verify your account has access to the requested partition
- Make sure output directories exist (`mkdir -p out/`)

## File Locations Summary

- **Enroot Image**: `/dss/dsshome1/lxc0E/go29sen2/multimodalneuralvideocompression+latest.sqsh`
- **Project Code**: Mounted from your DSS project path to `/workspace`
- **Data**: Mounted from your DSS data path to `/workspace/dataset`
- **Logs**: `out/video-compression-{job_id}.out` and `.err`