# ManiTaskGen Installation Instructions


This document provides instructions for installing and setting up the ManiTaskGen environment.



## Create and activate Conda environment

```shell
conda create -name manitaskgen python==3.10.16
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
cd ..
cd ai2thor_maniskill
python -m mani_skill.utils.download_asset AI2THOR
https://huggingface.co/datasets/haosulab/AI2THOR/tree/main
cp -r /path/to/.maniskill/data/datasets/ai2thor .

cd ..
cd replica_maniskill
python -m mani_skill.utils.download_asset ReplicaCAD
cp -r /path/to/.maniskill/data/datasets/replica_cad_dataset .
```

Note that the dataset will be downloaded to the subfolder of maniskill module, and the absolute path will be shown after the dataset downloading command is finished. 

After the loading the scene, change the dataset path in ``AppConfig``,`RawSceneConfig` and `SapienConfig` class in `src/utils/config_manager.py` accordingly.

If you have issues when downloading with maniskill, The datasets can also be downloaded from their repository on hugging-face website:

``https://huggingface.co/datasets/haosulab/AI2THOR``

``https://huggingface.co/datasets/haosulab/ReplicaCAD``



### SUNRGBD 

Download from the official website: ``https://rgbd.cs.princeton.edu/``

Unzip and place the dataset under the `data/dataset/sunrgbd` directory.

## Setting Up VLM APIs

We use OpenRouter API to access both open-source and closed-source VLMs. To benchmark VLM agents, you need to set up your OpenRouter API key and model address in the configuration file or command line arguments.

### Encode your OpenRouter API_key

Modify  ``OpenRouterConfig `` class in `src/utils/config_manager.py` (or your yml config file) with your API key.

For more details on using the OpenRouter API, refer to the OpenRouter  [official documentation](https://openrouter.ai/docs/quickstart).







