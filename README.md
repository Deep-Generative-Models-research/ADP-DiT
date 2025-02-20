# ADP-DiT 

ADPDiT: A Multimodal Diffusion Transformer for Predicting Alzheimer's Disease Progression and Generating Brain Image


## Requirements

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

## Dependencies and Installation

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

## Training

### Data Preparation

  Refer to the commands below to prepare the training data. 
  
  1. Install dependencies
  
      We offer an efficient data management library, named IndexKits, supporting the management of reading hundreds of millions of data during training, see more in [docs](./IndexKits/README.md).
      ```shell
      # 1 Install dependencies
      cd ADP-DiT
      pip install -e ./IndexKits
     ```
  2. Data download 
  
     ```shell
     # 2 Data download
     mkdir ./dataset/AD3_meta/arrows ./dataset/AD3_meta/jsons
     ```

     ```shell  
     # 3 Data conversion 
     python ./adpdit/data_loader/csv2arrow.py ./dataset/AD3_meta/csvfile/image_text_all.csv ./dataset/AD3_meta/arrows 1
     ```
  
  4. Data Selection and Configuration File Creation 
     
      We configure the training data through YAML files. In these files, you can set up standard data processing strategies for filtering, copying, deduplicating, and more regarding the training data. For more details, see [./IndexKits](IndexKits/docs/MakeDataset.md).
  
      For a sample file, please refer to [file](./dataset/yamls/porcelain.yaml). For a full parameter configuration file, see [file](./IndexKits/docs/MakeDataset.md).
  
     
  5. Create training data index file using YAML file.
    
     ```shell
      # Single Resolution Data Preparation
      idk base -c dataset/yamls/AD3_meta.yaml -t dataset/AD/jsons/AD3_meta.json
     
      ```
   
  The directory structure for `porcelain` dataset is:

  ```shell
   cd ./dataset
  
   porcelain
      ├──arrows/  (arrow files containing all necessary training data)
      │  ├──00000.arrow
      │  ├──00001.arrow\
      │  ├──......
      ├──csvfile/  (csv files containing text-image pairs)
      │  ├──image_text.csv
      ├──images/  (image files)
      │  ├──001_S_0001_2025-01-01_165.png
      │  ├──001_S_0001_2025-02-01_165.png
      │  ├──......
      ├──jsons/  (final training data index files which read data from arrow files during training)
      │  ├──AD3_meta.json
      │  ├──AD3_meta_stats.txt
   ```

### Training
  
  **Requirement:** 
  1. The minimum requriment is a single GPU with at least 24GB memory, but we recommend to use a GPU with about 30 GB memory to avoid host memory offloading. 
  2. Additionally, we encourage users to leverage the multiple GPUs across different nodes to speed up training on large datasets. 
  

  ```shell
  # Single Resolution Training
  export PYTHONPATH=/workspace/IndexKits:$PYTHONPATH
  sh adpdit/train.sh 

  ```

  After checkpoints are saved, you can use the following command to evaluate the model.
  ```shell
  # Inference
  python adpdit/infer_CNMCADtoCNMCAD.py

  ```
  
