
## Requirements

```
maniskill=3.0.0b15

numpy==1.26.4

shapely==2.0.6

sapien==3.0.0b1

transforms3d==0.4.2

trimesh==4.5.2

triangle==20230923

pinocchio==3.3.1

```



**You may need to use `` conda install pinocchio -c conda-forge`` to install pinocchio.**




To run:

Run ``python main.py`` can run the whole sample pipeline, including the following steps:

* call ``parse_replica.parse_replica(input_json_path, output_json_path)`` in ``scene_graph/parse_replica.py``, parse the original replica file, get a raw pose for every object.

* call ``visualize_scene_sapien.load_objects_from_json(scene, output_json_path), visualize_scene_sapien.reget_entities_from_sapien(scene, output_json_path, accurate_entity_path) `` in ``scene_graph/visualize_scene_sapien.py``. The first function loads all the objects into the sapien; the second function retrieve the objects' pose after step simulated in sapien.

* call ``gen_scene_graph.load_json_file(accurate_entity_path), gen_scene_graph.gen_multi_layer_graph_with_free_space(json_tree_path)`` in ``scene_graph/gen_scene_graph.py``.The first function loads all the objects into the sapien, the second function build scene graph tree according to the objects.

* create a ``TaskGeneration`` instance defined in ``atomic_task_generation.py``, and call according functions to generate atomic tasks.

* After this, sample some tasks and manually start interaction.

## functions of each important .py file

### scene_graph/parse_replica.py
Parse the pose of objects from raw replica file.

A list of deleted objects is as follows:

['frl_apartment_handbag', 'frl_apartment_cushion_01', 'frl_apartment_monitor', 'frl_apartment_cloth_01', 'frl_apartment_cloth_02', 'frl_apartment_cloth_03', 'frl_apartment_cloth', 'frl_apartment_umbrella',
'frl_apartment_tv_screen', 'frl_apartment_indoor_plant_01', 'frl_apartment_monitor_stand', 'frl_apartment_setupbox', 'frl_apartment_beanbag', 'frl_apartment_bike_01', 'frl_apartment_bike_02', 'frl_apartment_indoor_plant_02', 'frl_apartment_picture_01', 'frl_apartment_towel', 'frl_apartment_rug_01', 'frl_apartment_rug_02', 'frl_apartment_rug_03', 'frl_apartment_mat',
                           ]

Reason of deleting items consists:

* Some ground items, such as the bike and beanbag, are not likely to be a sensible destination for the task. 

* Mats and rugs often lies under the table, make table doesn't contact with ground directly, interfere the direction logic of ground objects.

* Hangables are deleted, including a kitchen wall cabinet above the counter.

* Some small items fell to the ground after we delete all kinds of other not desired objects, And we don't want to put items on them.

### scene_graph/visualize_scene_sapien.py

Simulate the scene in sapien after receiving the output from parse_replica.py.

Since the initial pose often have a little offset, we need to retrieve more accurate entities' pose from it after enough steps.

### scene_graph/gen_scene_graph.py

Generate the scene graph. 

Use ``load_json_file(accurate_entity_path)`` function to load the retrieved entities' pose, and use ``gen_multi_layer_graph_with_free_space(json_tree_path)`` to generate a scene graph, The crucial structure of our work.

#### auxiliary files:

* In ``scene_graph/object`` there are a few ``.py`` files supporting the construct of scene graph. 

** ``scene_graph/object/parse_object.py`` create an


### atomic_task_generation/atomic_task_generation.py

### render_image/imagepoint.py

### render_image/sapienprocessor.py




  
