# Explanation for outcome-based task generation. 

This document explains all the possible patterns in the ManiTaskOT-200, and how they are used for outcome-based task generation.

## Object Definitions

### Ground Objects
Ground objects are the primary objects in the environment that can be interacted with. In the scene graph, they will have depth=1. They include:

* Single Layer Objects: len(node.own_platform) = 1
* Multi Layer Objects: len(node.own_platform) > 1

In the task description, they will be referred as receptacles with different patterns. 

### Surface Objects 
Surface objects are objects that can be placed on top of ground objects. In the scene graph, they will have depth>1. 

They will have a category name after item renaming. Their category names are the first word in their original names. For example, a shoe will have name "furniture_shoe_rack_shelf_black_shoes". And this will used for indexing. 



### 


## Patterns
The outcome-based tasks in ManiTaskOT-200 are generated using a variety of patterns that define the structure and components of each task. Below are the key patterns used:



### Platform Patterns 

**[PlatformX]**: This pattern represents the platform number X where the task is to be performed. For example, [Platform1] indicates that the task is to be executed on Platform 1. It's also guaranteed that if the task involves multiple platforms, they will not belong to the same ground object.

**[Multilayer-ObjectX]**: This pattern represents an object that consists of multiple layers, each potentially having different properties and interactions. Currently there are at most one multi-layer object in a single task. 

There will also be tasks that involves specific layers of the multi-layer object. 

* [ALL-LAYERS]: This indicates that the task involves all layers of the multi-layer object.
* [SPECIFIC-LAYER]: This indicates that the task involves a specific layer of the multi-layer object.
* [TOP-LAYER]: This indicates that the task involves the top layer of the multi-layer object.

### Surface Object Patterns 

This pattern will be used to to indicate the surface object that will be used in the task.

**[SUB-PLATFORM-CATEGORY-OBJECTXY]**: This pattern indicates a category of surface objects (e.g. all books, all kitchenwares) that is located on a sub-platform of the main platform.  The "X" represents the platform number, and "Y" represents the index of the surface object on that platform. It is guaranteed that there will be a [PLATFORMX] in the same task.

**[SUB-PLATFORM-OBJECTXY]**: This pattern indicates a kind of specific surface object (e.g. all blue books) that is located on a sub-platform of the main platform.  The "X" represents the platform number, and "Y" represents the index of the surface object on that platform. It is guaranteed that there will be a [PLATFORMX] in the same task.

**[SUB-PLATFORM-SINGLE-OBJECTXY]**: This pattern indicates a single specific surface object (e.g. one blue book) that is located on a sub-platform of the main platform.  The "X" represents the platform number, and "Y" represents the index of the surface object on that platform. It is guaranteed that there will be a [PLATFORMX] in the same task.

**[SUB-LAYER-CATEGORY-OBJECTX]**: This pattern indicates a category of surface objects (e.g. all books, all kitchenwares on a kitchen counter) that is located in a multi-layer object. The "X" represents the index of the surface object on that layer. It is guaranteed that there will be a [MULTILAYER-OBJECT]X in the same task.

**[SUB-LAYER-OBJECTX]**: This pattern indicates a kind of specific surface object (e.g. all blue books on a wall cabinet) that is located in a multi-layer object. The "X" represents the index of the surface object on that layer. It is guaranteed that there will be a [MULTILAYER-OBJECTX] in the same task.


* 