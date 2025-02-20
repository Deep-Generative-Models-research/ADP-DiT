# ADP-DiT 

ADPDiT: A Multimodal Diffusion Transformer for Predicting Alzheimer's Disease Progression and Generating Brain Image


## 📜 Requirements

This repo consists of ADP-DiT (text guided image-to-image model).

The following table shows the requirements for running the models

|          Model          |      Training Time      |   GPU  Memory   |       GPU       |
|:-----------------------:|:-----------------------:|:---------------:|:---------------:|
|         ADP-DiT         |          3weeks         |       20G*8    |    4000ada*8    |

* An NVIDIA GPU with CUDA support is required. 
  * We have tested 4000ada*8 GPUs.
  * **Minimum**: The minimum GPU memory required is 12GB.
  * **Recommended**: We recommend using a GPU with 24GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Deep-Generative-Models-research/ADP-DiT.git
cd ADP-DiT
```

### Installation Guide for Linux

We provide an `requirements.txt` file for setting up a python venv.
python venv's installation instructions are available [here](https://docs.python.org/ko/3.10/library/venv.html).

We recommend CUDA versions 12.4+ and python 3.10.12 version.

```shell
# 1. Prepare python venv environment
python3 -m venv venv

# 2. Activate the environment
source venv/bin/activate

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. Install flash attention v2 for acceleration (requires CUDA 12.4 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

Additionally, you can also use docker to set up the environment.
```shell
# 1. Use the following link to download the docker image tar file.
# For CUDA 12 (un upload)
wget https://~~~~~~~/adp_dit_cu12.tar


# 2. Import the docker tar file and show the image meta information
# For CUDA 12
docker load -i adp_dit_cu12.tar


docker image ls

# 3. Run the container based on the image
docker run --gpus all -it --shm-size=8g -v your repository:/workspace -p 8888:8888 adp-dit:cuda12
```

  
