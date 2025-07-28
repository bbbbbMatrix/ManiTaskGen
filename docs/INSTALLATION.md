# ManiTaskGen Installation Instructions


This document provides instructions for installing and setting up the ManiTaskGen environment.



## Create and activate Conda environment

```shell
conda create -name manitaskgen python==3.10.16
conda activate manitaskgen
```



## Install dependencies and requirements

```shell
conda install pinocchio -c conda-forge
conda install pytorch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```

If you have issues with library linking make sure that the conda libraries are in your LD_LIBRARY_PATH (e.g `export LD_LIBRARY_PATH=/path/to/anaconda/envs/myenv/lib:$LD_LIBRARY_PATH`).



## Download datasets



### AI2THOR and ReplicaCAD



```shell
cd ..
cd ai2thor_maniskill
python -m mani_skill.utils.download_asset AI2THOR
https://huggingface.co/datasets/haosulab/AI2THOR/tree/main
cp -r /path/to/.maniskill/data/scene_datasets/ai2thor .

cd ..
cd replica_maniskill
python -m mani_skill.utils.download_asset ReplicaCAD
cp -r /path/to/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange .
```

Note that the dataset will be downloaded to the subfolder of maniskill module, and the absolute path will be shown after the dataset downloading command is finished. 

After the loading the scene, change the dataset path in ``AppConfig``,`RawSceneConfig` and `SapienConfig` class in `src/utils/config_manager.py` accordingly.

If you have issues when downloading with maniskill, The datasets can also be downloaded from their repository on hugging-face website:

``https://huggingface.co/datasets/haosulab/AI2THOR``

``https://huggingface.co/datasets/haosulab/ReplicaCAD``



### SUNRGBD 

Download from the official website: ``https://rgbd.cs.princeton.edu/``

Though unable to benchmark, you can run the following command to parse the sunrgbd scene, and get a json file for building scene graph and generating tasks:

```shell
python src/preprocessing/sunrgbd_parser.py --scene_path=path/to/SUNRGBD/dataset --output_path=/path/to/output/folder
```







