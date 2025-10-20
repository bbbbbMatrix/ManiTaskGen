# ManiTaskGen Installation Instructions


This document provides instructions for installing and setting up the ManiTaskGen environment.



## Create and activate Conda environment

```shell
conda create -n manitaskgen python=3.10.16
conda activate manitaskgen
```

## Clone the repository

```shell
git clone https://github.com/bbbbbMatrix/ManiTaskGen.git
cd ManiTaskGen
cd src/vlm_interaction
git clone https://github.com/ManiacWallnut/VLMEvalKit.git
cd ../..
```

## Install dependencies and requirements

```shell
bash install.sh
```




## Download datasets



### AI2THOR and ReplicaCAD



```shell
cd data 
mkdir datasets
cd datasets

python -m mani_skill.utils.download_asset AI2THOR
cp -r /path/to/.maniskill/data/scene_datasets/ai2thor .

python -m mani_skill.utils.download_asset ReplicaCAD
cp -r /path/to/.maniskill/data/scene_datasets/replica_cad_dataset .
cd ../..
```

Note that the dataset will be downloaded to the subfolder of maniskill module, and the absolute path will be shown after the dataset downloading command is finished. 


If you have issues when downloading with maniskill, The datasets can also be downloaded from their repository on hugging-face website:

``https://huggingface.co/datasets/haosulab/AI2THOR``

``https://huggingface.co/datasets/haosulab/ReplicaCAD``


### SUNRGBD

Download from the official website: https://rgbd.cs.princeton.edu/

## Setting Up VLM APIs

We use [OpenRouter](https://openrouter.ai/) API to access both open-source and closed-source VLMs. To benchmark VLM agents, you need to set up your OpenRouter API key and model address in the configuration file, i.e. modify config in the ``common:open_router:model`` column with a valid model address.  See Readme for more details.

### Encode your OpenRouter API_key

Modify config in the ``common:open_router:api_key`` column with your API key.  See Readme for more details.

For more details on using the OpenRouter API, refer to the OpenRouter  [official documentation](https://openrouter.ai/docs/quickstart).









