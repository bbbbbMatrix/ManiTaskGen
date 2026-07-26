INPUT_STAGE = SCENE_BATCH
SCENE_ID = apt_0
BATCH_ID = 1/1

COORDINATE_CONVENTION:
- units: meters
- axes: x, y in the ground plane; z up. Origin = world origin of the source scene.
- position: 3D centroid {x, y, z}.
- orientation:
    heading = {hx, hy}  # 2D forward unit vector in the ground plane (hx=cos, hy=sin)
    quaternion = {w, x, y, z}  # full 3D orientation from the source scene
- bounding box:
    bbox_2d = [{x, y} x 4]  # heading-aligned rectangle, 4 corners, CCW
    z_range = [z_min, z_max]  # vertical extent
- direction convention (relative placement / freespace), 0-based, CCW from rear,
  verbatim from the scene-graph code constants EIGHT_DIRECTIONS / NINE_DIRECTIONS:
    0=rear, 1=rear-left, 2=left, 3=front-left, 4=front, 5=front-right, 6=right,
    7=rear-right; 8=center.

RAW_OBJECT_DATA:
{
  "platforms": [
  {
    "platform_id": "frl_apartment_sofa_10_0",
    "name": "sofa_10_platform_0",
    "base_object": "frl_apartment_sofa_10",
    "centroid": {
      "x": 3.8342,
      "y": -5.2859,
      "z": 0.2901
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 4.2474,
        "y": -6.1865
      },
      {
        "x": 3.4198,
        "y": -6.1859
      },
      {
        "x": 3.4211,
        "y": -4.3853
      },
      {
        "x": 4.2486,
        "y": -4.3858
      }
    ],
    "z_range": [
      0.2901,
      2.7901
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_sofa_10_1",
    "name": "sofa_10_platform_1",
    "base_object": "frl_apartment_sofa_10",
    "centroid": {
      "x": 3.7109,
      "y": -5.7269,
      "z": 0.4132
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 4.0138,
        "y": -6.1537
      },
      {
        "x": 3.4073,
        "y": -6.1533
      },
      {
        "x": 3.4079,
        "y": -5.3
      },
      {
        "x": 4.0144,
        "y": -5.3005
      }
    ],
    "z_range": [
      0.4132,
      2.9132
    ],
    "children_objects": [
      "furniture_light_blue_square_cushion_on_sofa_platform_11",
      "furniture_light_blue_square_cushion_on_sofa_platform_12"
    ]
  },
  {
    "platform_id": "frl_apartment_sofa_10_2",
    "name": "sofa_10_platform_2",
    "base_object": "frl_apartment_sofa_10",
    "centroid": {
      "x": 3.7088,
      "y": -4.8215,
      "z": 0.4132
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 4.0118,
        "y": -5.2483
      },
      {
        "x": 3.4053,
        "y": -5.2479
      },
      {
        "x": 3.4059,
        "y": -4.3946
      },
      {
        "x": 4.0123,
        "y": -4.395
      }
    ],
    "z_range": [
      0.4132,
      2.9132
    ],
    "children_objects": [
      "furniture_light_blue_square_cushion_on_sofa_platform_9"
    ]
  },
  {
    "platform_id": "frl_apartment_shoe_04_80_0",
    "name": "shoe_04_80_platform_0",
    "base_object": "frl_apartment_shoe_04_80",
    "centroid": {
      "x": -1.5854,
      "y": 2.859,
      "z": 0.0378
    },
    "heading": {
      "hx": 0.0179,
      "hy": 0.9998
    },
    "quaternion": {
      "w": 0.0385,
      "x": -0.0025,
      "y": 0.0106,
      "z": 0.9992
    },
    "bbox_2d": [
      {
        "x": -1.5379,
        "y": 2.7895
      },
      {
        "x": -1.6355,
        "y": 2.7913
      },
      {
        "x": -1.633,
        "y": 2.9284
      },
      {
        "x": -1.5354,
        "y": 2.9267
      }
    ],
    "z_range": [
      0.0378,
      2.5378
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_shoe_04_80_1",
    "name": "shoe_04_80_platform_1",
    "base_object": "frl_apartment_shoe_04_80",
    "centroid": {
      "x": -1.7367,
      "y": 2.8614,
      "z": 0.0401
    },
    "heading": {
      "hx": 0.0179,
      "hy": 0.9998
    },
    "quaternion": {
      "w": 0.0385,
      "x": -0.0025,
      "y": 0.0106,
      "z": 0.9992
    },
    "bbox_2d": [
      {
        "x": -1.6907,
        "y": 2.792
      },
      {
        "x": -1.7853,
        "y": 2.7937
      },
      {
        "x": -1.7828,
        "y": 2.9307
      },
      {
        "x": -1.6882,
        "y": 2.929
      }
    ],
    "z_range": [
      0.0401,
      2.5401
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_table_04_13_0",
    "name": "table_04_13_platform_0",
    "base_object": "frl_apartment_table_04_13",
    "centroid": {
      "x": 4.1999,
      "y": -6.6244,
      "z": 0.494
    },
    "heading": {
      "hx": 0.1564,
      "hy": -0.9877
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": 3.9339,
        "y": -6.4311
      },
      {
        "x": 4.3932,
        "y": -6.3584
      },
      {
        "x": 4.466,
        "y": -6.8177
      },
      {
        "x": 4.0066,
        "y": -6.8905
      }
    ],
    "z_range": [
      0.494,
      2.994
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_04_46_0",
    "name": "chair_04_46_platform_0",
    "base_object": "frl_apartment_chair_04_46",
    "centroid": {
      "x": -2.1398,
      "y": -3.847,
      "z": 0.429
    },
    "heading": {
      "hx": -0.9996,
      "hy": 0.0284
    },
    "quaternion": {
      "w": 0.7071,
      "x": -0.0,
      "y": 0.0,
      "z": -0.7071
    },
    "bbox_2d": [
      {
        "x": -1.9377,
        "y": -3.617
      },
      {
        "x": -1.9511,
        "y": -4.0882
      },
      {
        "x": -2.3419,
        "y": -4.0771
      },
      {
        "x": -2.3286,
        "y": -3.6059
      }
    ],
    "z_range": [
      0.429,
      2.929
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_05_8_0",
    "name": "chair_05_8_platform_0",
    "base_object": "frl_apartment_chair_05_8",
    "centroid": {
      "x": 4.0357,
      "y": -0.0867,
      "z": 0.6503
    },
    "heading": {
      "hx": 1.0,
      "hy": -0.0066
    },
    "quaternion": {
      "w": 0.7064,
      "x": 0.0,
      "y": 0.0,
      "z": 0.7078
    },
    "bbox_2d": [
      {
        "x": 3.8791,
        "y": -0.2833
      },
      {
        "x": 3.8818,
        "y": 0.112
      },
      {
        "x": 4.1922,
        "y": 0.1099
      },
      {
        "x": 4.1896,
        "y": -0.2854
      }
    ],
    "z_range": [
      0.6503,
      3.1503
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_05_7_0",
    "name": "chair_05_7_platform_0",
    "base_object": "frl_apartment_chair_05_7",
    "centroid": {
      "x": 3.9919,
      "y": -0.7357,
      "z": 0.6503
    },
    "heading": {
      "hx": 1.0,
      "hy": -0.0096
    },
    "quaternion": {
      "w": 0.7075,
      "x": 0.0,
      "y": 0.0,
      "z": 0.7067
    },
    "bbox_2d": [
      {
        "x": 3.8348,
        "y": -0.9319
      },
      {
        "x": 3.8386,
        "y": -0.5366
      },
      {
        "x": 4.1491,
        "y": -0.5396
      },
      {
        "x": 4.1453,
        "y": -0.9349
      }
    ],
    "z_range": [
      0.6503,
      3.1503
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_stool_02_18_0",
    "name": "stool_02_18_platform_0",
    "base_object": "frl_apartment_stool_02_18",
    "centroid": {
      "x": 0.5558,
      "y": -7.7785,
      "z": 0.4135
    },
    "heading": {
      "hx": -1.0,
      "hy": -0.0003
    },
    "quaternion": {
      "w": 0.7069,
      "x": -0.0,
      "y": 0.0,
      "z": -0.7073
    },
    "bbox_2d": [
      {
        "x": 0.7565,
        "y": -7.5791
      },
      {
        "x": 0.7566,
        "y": -7.9778
      },
      {
        "x": 0.355,
        "y": -7.9779
      },
      {
        "x": 0.3549,
        "y": -7.5792
      }
    ],
    "z_range": [
      0.4135,
      2.9135
    ],
    "children_objects": [
      "furniture_gray_tissue_box_on_platform_17"
    ]
  },
  {
    "platform_id": "frl_apartment_stool_02_6_0",
    "name": "stool_02_6_platform_0",
    "base_object": "frl_apartment_stool_02_6",
    "centroid": {
      "x": 4.0363,
      "y": -2.371,
      "z": 0.4135
    },
    "heading": {
      "hx": -0.9978,
      "hy": 0.0663
    },
    "quaternion": {
      "w": 0.7304,
      "x": -0.0,
      "y": 0.0,
      "z": 0.683
    },
    "bbox_2d": [
      {
        "x": 4.2499,
        "y": -2.1854
      },
      {
        "x": 4.2235,
        "y": -2.5832
      },
      {
        "x": 3.8228,
        "y": -2.5566
      },
      {
        "x": 3.8492,
        "y": -2.1587
      }
    ],
    "z_range": [
      0.4135,
      2.9135
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_rack_01_76_0",
    "name": "rack_01_76_platform_0",
    "base_object": "frl_apartment_rack_01_76",
    "centroid": {
      "x": -1.9758,
      "y": 2.6184,
      "z": 0.0913
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": {
      "w": 0.9974,
      "x": 0.0021,
      "y": -0.0003,
      "z": -0.0714
    },
    "bbox_2d": [
      {
        "x": -1.5572,
        "y": 2.7111
      },
      {
        "x": -1.6028,
        "y": 2.407
      },
      {
        "x": -2.3945,
        "y": 2.5257
      },
      {
        "x": -2.3488,
        "y": 2.8298
      }
    ],
    "z_range": [
      0.0913,
      0.2868
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_rack_01_76_1",
    "name": "rack_01_76_platform_1",
    "base_object": "frl_apartment_rack_01_76",
    "centroid": {
      "x": -1.976,
      "y": 2.6175,
      "z": 0.2987
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": {
      "w": 0.9974,
      "x": 0.0021,
      "y": -0.0003,
      "z": -0.0714
    },
    "bbox_2d": [
      {
        "x": -1.5574,
        "y": 2.7102
      },
      {
        "x": -1.603,
        "y": 2.4061
      },
      {
        "x": -2.3947,
        "y": 2.5248
      },
      {
        "x": -2.349,
        "y": 2.8289
      }
    ],
    "z_range": [
      0.2987,
      0.4465
    ],
    "children_objects": [
      "footwear_red_high_heel_on_shelf_platform_79",
      "footwear_black_sneaker_on_platform_77",
      "footwear_brown_leather_dress_shoe_on_platform_78"
    ]
  },
  {
    "platform_id": "frl_apartment_rack_01_76_2",
    "name": "rack_01_76_platform_2",
    "base_object": "frl_apartment_rack_01_76",
    "centroid": {
      "x": -1.9762,
      "y": 2.6167,
      "z": 0.4842
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": {
      "w": 0.9974,
      "x": 0.0021,
      "y": -0.0003,
      "z": -0.0714
    },
    "bbox_2d": [
      {
        "x": -1.5576,
        "y": 2.7094
      },
      {
        "x": -1.6032,
        "y": 2.4053
      },
      {
        "x": -2.3948,
        "y": 2.524
      },
      {
        "x": -2.3492,
        "y": 2.8281
      }
    ],
    "z_range": [
      0.4842,
      2.9842
    ],
    "children_objects": []
  },
  {
    "platform_id": "kitchen_counter_1_body_0",
    "name": "kitchen_counter_1_body_platform_0",
    "base_object": "kitchen_counter_1_body",
    "centroid": {
      "x": -2.164,
      "y": -1.26,
      "z": 0.0891
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.8557,
        "y": -2.7467
      },
      {
        "x": -2.4427,
        "y": -2.7525
      },
      {
        "x": -2.4723,
        "y": 0.2266
      },
      {
        "x": -1.8853,
        "y": 0.2325
      }
    ],
    "z_range": [
      0.0891,
      0.8027
    ],
    "children_objects": [
      "electronics_black_wall_bracket_on_platform_2"
    ]
  },
  {
    "platform_id": "kitchen_counter_1_body_1",
    "name": "kitchen_counter_1_body_platform_1",
    "base_object": "kitchen_counter_1_body",
    "centroid": {
      "x": -2.1811,
      "y": -0.1926,
      "z": 0.6874
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.9184,
        "y": -0.6136
      },
      {
        "x": -2.4353,
        "y": -0.6188
      },
      {
        "x": -2.4437,
        "y": 0.2284
      },
      {
        "x": -1.9269,
        "y": 0.2335
      }
    ],
    "z_range": [
      0.6874,
      0.7918
    ],
    "children_objects": []
  },
  {
    "platform_id": "kitchen_counter_1_body_2",
    "name": "kitchen_counter_1_body_platform_2",
    "base_object": "kitchen_counter_1_body",
    "centroid": {
      "x": -2.1716,
      "y": -1.2649,
      "z": 0.8615
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.8432,
        "y": -2.8095
      },
      {
        "x": -2.4692,
        "y": -2.8157
      },
      {
        "x": -2.4999,
        "y": 0.2797
      },
      {
        "x": -1.8739,
        "y": 0.2859
      }
    ],
    "z_range": [
      0.8615,
      3.3615
    ],
    "children_objects": [
      "kitchenware_white_cylindrical_spice_jars_on_platform_59",
      "kitchenware_orange_spice_shaker_on_white_platform_66",
      "kitchenware_orange_spice_shaker_on_white_platform_67",
      "kitchenware_orange_spice_shaker_on_white_platform_65",
      "kitchenware_orange_spice_shaker_on_white_platform_64",
      "kitchenware_orange_spice_shaker_on_white_platform_62",
      "kitchenware_orange_spice_shaker_on_white_platform_60",
      "kitchenware_orange_spice_shaker_on_white_platform_61",
      "kitchenware_orange_spice_shaker_on_white_platform_63",
      "kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
      "decor_photo_frame_with_dog_picture_on_platform_70",
      "kitchenware_white_octagonal_plate_on_black_platform_50",
      "kitchenware_beige_coffee_cup_on_black_table_platform_51",
      "kitchenware_white_flower_patterned_cream_jug_on_platform_55",
      "kitchenware_white_shallow_bowl_on_table_platform_57",
      "kitchenware_white_small_coffee_mug_on_black_table_platform_52",
      "kitchenware_gray_cylindrical_container_on_platform_72",
      "kitchenware_blue_thermos_bottle_73",
      "kitchenware_white_round_bowl_on_black_table_platform_54",
      "kitchenware_transparent_bowl_on_platform_56",
      "kitchenware_wooden_knife_block_on_platform_71",
      "furniture_brown_cushion_on_platform_53",
      "kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
      "container_brown_paper_food_box_on_table_platform_0",
      "kitchenware_white_cylindrical_cup_on_black_platform_68",
      "kitchenware_brown_black_coffee_grinder_on_table_platform_58"
    ]
  },
  {
    "platform_id": "fridge_0_body_1",
    "name": "fridge_0_body_platform_1",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.1758,
      "y": -3.2422,
      "z": 0.041
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.9333,
        "y": -3.5405
      },
      {
        "x": -2.4223,
        "y": -3.5372
      },
      {
        "x": -2.4183,
        "y": -2.9439
      },
      {
        "x": -1.9293,
        "y": -2.9471
      }
    ],
    "z_range": [
      0.041,
      0.6147
    ],
    "children_objects": []
  },
  {
    "platform_id": "fridge_0_body_3",
    "name": "fridge_0_body_platform_3",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.2196,
      "y": -3.2386,
      "z": 0.6469
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.9734,
        "y": -3.5561
      },
      {
        "x": -2.47,
        "y": -3.5528
      },
      {
        "x": -2.4658,
        "y": -2.9212
      },
      {
        "x": -1.9691,
        "y": -2.9245
      }
    ],
    "z_range": [
      0.6469,
      0.9955
    ],
    "children_objects": []
  },
  {
    "platform_id": "fridge_0_body_4",
    "name": "fridge_0_body_platform_4",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.2085,
      "y": -3.2366,
      "z": 1.0015
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.9236,
        "y": -3.5657
      },
      {
        "x": -2.4979,
        "y": -3.5618
      },
      {
        "x": -2.4935,
        "y": -2.9075
      },
      {
        "x": -1.9192,
        "y": -2.9114
      }
    ],
    "z_range": [
      1.0015,
      1.1781
    ],
    "children_objects": []
  },
  {
    "platform_id": "fridge_0_body_5",
    "name": "fridge_0_body_platform_5",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.1437,
      "y": -3.2471,
      "z": 1.1816
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.9241,
        "y": -3.5675
      },
      {
        "x": -2.3677,
        "y": -3.5645
      },
      {
        "x": -2.3634,
        "y": -2.9268
      },
      {
        "x": -1.9198,
        "y": -2.9297
      }
    ],
    "z_range": [
      1.1816,
      1.5503
    ],
    "children_objects": []
  },
  {
    "platform_id": "fridge_0_body_6",
    "name": "fridge_0_body_platform_6",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.1356,
      "y": -3.2421,
      "z": 1.5498
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.8919,
        "y": -3.4947
      },
      {
        "x": -2.3826,
        "y": -3.4914
      },
      {
        "x": -2.3793,
        "y": -2.9895
      },
      {
        "x": -1.8885,
        "y": -2.9928
      }
    ],
    "z_range": [
      1.5498,
      1.8988
    ],
    "children_objects": []
  },
  {
    "platform_id": "fridge_0_body_7",
    "name": "fridge_0_body_platform_7",
    "base_object": "fridge_0_body",
    "centroid": {
      "x": -2.1799,
      "y": -3.2273,
      "z": 1.9222
    },
    "heading": {
      "hx": 0.0067,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7092,
      "x": 0.7051,
      "y": 0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -1.861,
        "y": -3.5759
      },
      {
        "x": -2.5034,
        "y": -3.5716
      },
      {
        "x": -2.4988,
        "y": -2.8787
      },
      {
        "x": -1.8564,
        "y": -2.883
      }
    ],
    "z_range": [
      1.9222,
      4.4222
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_bin_03_3_0",
    "name": "bin_03_3_platform_0",
    "base_object": "frl_apartment_bin_03_3",
    "centroid": {
      "x": 4.1815,
      "y": -7.1107,
      "z": 0.6157
    },
    "heading": {
      "hx": -0.0103,
      "hy": -0.9999
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": 4.0227,
        "y": -6.8811
      },
      {
        "x": 4.345,
        "y": -6.8844
      },
      {
        "x": 4.3403,
        "y": -7.3402
      },
      {
        "x": 4.018,
        "y": -7.3369
      }
    ],
    "z_range": [
      0.6157,
      3.1157
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_0",
    "name": "wall_cabinet_01_4_platform_0",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9466,
      "z": 0.1629
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0271,
        "y": -4.1647
      },
      {
        "x": 4.0159,
        "y": -3.7362
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      0.1629,
      0.4922
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_1",
    "name": "wall_cabinet_01_4_platform_1",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9466,
      "z": 0.5054
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0272,
        "y": -4.1647
      },
      {
        "x": 4.0159,
        "y": -3.7362
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      0.5054,
      0.7809
    ],
    "children_objects": [
      "decor_brown_wooden_mantle_clock_on_wall_shelf_5"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_2",
    "name": "wall_cabinet_01_4_platform_2",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9466,
      "z": 0.794
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0272,
        "y": -4.1647
      },
      {
        "x": 4.0159,
        "y": -3.7362
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      0.794,
      1.0801
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_3",
    "name": "wall_cabinet_01_4_platform_3",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9466,
      "z": 1.0933
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0272,
        "y": -4.1647
      },
      {
        "x": 4.0159,
        "y": -3.7362
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      1.0933,
      1.3523
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_4",
    "name": "wall_cabinet_01_4_platform_4",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9466,
      "z": 1.3655
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0272,
        "y": -4.1647
      },
      {
        "x": 4.0159,
        "y": -3.7362
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      1.3655,
      1.6386
    ],
    "children_objects": [
      "book_maroon_thin_hardcover_on_white_shelf_platform_88",
      "book_maroon_thin_hardcover_on_white_shelf_platform_87",
      "book_maroon_thin_hardcover_on_white_shelf_platform_86"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_5",
    "name": "wall_cabinet_01_4_platform_5",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9465,
      "z": 1.6518
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.3204,
        "y": -4.157
      },
      {
        "x": 4.0272,
        "y": -4.1646
      },
      {
        "x": 4.0159,
        "y": -3.7361
      },
      {
        "x": 4.3092,
        "y": -3.7285
      }
    ],
    "z_range": [
      1.6518,
      1.9516
    ],
    "children_objects": [
      "container_brown_wooden_box_on_white_shelf_84"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_01_4_6",
    "name": "wall_cabinet_01_4_platform_6",
    "base_object": "frl_apartment_wall_cabinet_01_4",
    "centroid": {
      "x": 4.1682,
      "y": -3.9465,
      "z": 1.9669
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": {
      "w": 0.6937,
      "x": -0.0,
      "y": 0.0,
      "z": 0.7203
    },
    "bbox_2d": [
      {
        "x": 4.362,
        "y": -4.189
      },
      {
        "x": 3.9873,
        "y": -4.1988
      },
      {
        "x": 3.9744,
        "y": -3.704
      },
      {
        "x": 4.349,
        "y": -3.6942
      }
    ],
    "z_range": [
      1.9669,
      4.4669
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_table_03_14_1",
    "name": "table_03_14_platform_1",
    "base_object": "frl_apartment_table_03_14",
    "centroid": {
      "x": 1.99,
      "y": -5.8209,
      "z": 0.403
    },
    "heading": {
      "hx": -0.8045,
      "hy": 0.594
    },
    "quaternion": {
      "w": 0.95,
      "x": 0.0,
      "y": 0.0,
      "z": -0.3123
    },
    "bbox_2d": [
      {
        "x": 2.6856,
        "y": -5.8944
      },
      {
        "x": 2.2649,
        "y": -6.4641
      },
      {
        "x": 1.2943,
        "y": -5.7474
      },
      {
        "x": 1.715,
        "y": -5.1777
      }
    ],
    "z_range": [
      0.403,
      2.903
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_table_01_48_0",
    "name": "table_01_48_platform_0",
    "base_object": "frl_apartment_table_01_48",
    "centroid": {
      "x": 0.4144,
      "y": 0.1747,
      "z": 0.7484
    },
    "heading": {
      "hx": -1.0,
      "hy": 0.0006
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": 1.114,
        "y": 0.5998
      },
      {
        "x": 1.1135,
        "y": -0.2512
      },
      {
        "x": -0.2852,
        "y": -0.2504
      },
      {
        "x": -0.2847,
        "y": 0.6006
      }
    ],
    "z_range": [
      0.7484,
      3.2484
    ],
    "children_objects": [
      "electronics_white_security_camera_on_table_platform_49"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_0",
    "name": "wall_cabinet_02_21_platform_0",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4081,
      "z": 0.1654
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.8439
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      0.1654,
      0.4981
    ],
    "children_objects": [
      "book_black_thin_softcover_on_table_platform_36",
      "book_black_thin_softcover_on_table_platform_37",
      "kitchenware_brown_handle_knife_on_white_table_platform_38",
      "book_blue_thick_hardcover_on_white_shelf_platform_35"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_1",
    "name": "wall_cabinet_02_21_platform_1",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4081,
      "z": 0.5114
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.844
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      0.5114,
      0.7898
    ],
    "children_objects": [
      "book_white_thick_hardcover_on_table_platform_33",
      "book_maroon_thin_hardcover_on_white_shelf_platform_32",
      "book_navy_thick_hardcover_on_platform_34"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_2",
    "name": "wall_cabinet_02_21_platform_2",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4081,
      "z": 0.8031
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.844
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      0.8031,
      1.0922
    ],
    "children_objects": [
      "container_brown_wooden_box_on_white_shelf_85"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_3",
    "name": "wall_cabinet_02_21_platform_3",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4081,
      "z": 1.1056
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.844
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      1.1056,
      1.3673
    ],
    "children_objects": [
      "book_black_thin_softcover_on_table_platform_25",
      "kitchenware_brown_handle_knife_on_white_table_platform_39",
      "decor_abstract_colorful_painting_on_white_shelf_platform_22",
      "kitchenware_brown_handle_knife_on_white_table_platform_24",
      "book_navy_thick_hardcover_on_platform_23",
      "book_white_thick_hardcover_on_table_platform_29",
      "book_maroon_thin_hardcover_on_white_shelf_platform_31",
      "book_maroon_thin_hardcover_on_white_shelf_platform_28",
      "book_maroon_thin_hardcover_on_white_shelf_platform_30"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_4",
    "name": "wall_cabinet_02_21_platform_4",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4081,
      "z": 1.3806
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.844
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      1.3806,
      1.6567
    ],
    "children_objects": [
      "book_white_thick_hardcover_on_table_platform_27",
      "book_blue_thick_hardcover_on_white_shelf_platform_26"
    ]
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_5",
    "name": "wall_cabinet_02_21_platform_5",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4082,
      "z": 1.67
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.6939,
        "y": -5.8445
      },
      {
        "x": 0.396,
        "y": -5.844
      },
      {
        "x": 0.3975,
        "y": -4.9718
      },
      {
        "x": 0.6954,
        "y": -4.9723
      }
    ],
    "z_range": [
      1.67,
      1.9729
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_wall_cabinet_02_21_6",
    "name": "wall_cabinet_02_21_platform_6",
    "base_object": "frl_apartment_wall_cabinet_02_21",
    "centroid": {
      "x": 0.5457,
      "y": -5.4082,
      "z": 1.9884
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": {
      "w": 0.7071,
      "x": 0.0,
      "y": -0.0,
      "z": 0.7071
    },
    "bbox_2d": [
      {
        "x": 0.7341,
        "y": -5.8791
      },
      {
        "x": 0.3557,
        "y": -5.8785
      },
      {
        "x": 0.3573,
        "y": -4.9372
      },
      {
        "x": 0.7357,
        "y": -4.9378
      }
    ],
    "z_range": [
      1.9884,
      4.4884
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_01_15_1",
    "name": "chair_01_15_platform_1",
    "base_object": "frl_apartment_chair_01_15",
    "centroid": {
      "x": 1.9744,
      "y": -7.4666,
      "z": 0.3877
    },
    "heading": {
      "hx": -0.9994,
      "hy": 0.0353
    },
    "quaternion": {
      "w": 0.9998,
      "x": -0.0,
      "y": -0.0,
      "z": -0.0191
    },
    "bbox_2d": [
      {
        "x": 2.2347,
        "y": -7.2011
      },
      {
        "x": 2.2153,
        "y": -7.7499
      },
      {
        "x": 1.7141,
        "y": -7.7322
      },
      {
        "x": 1.7335,
        "y": -7.1834
      }
    ],
    "z_range": [
      0.3877,
      2.8877
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_01_15_2",
    "name": "chair_01_15_platform_2",
    "base_object": "frl_apartment_chair_01_15",
    "centroid": {
      "x": 1.9711,
      "y": -7.5455,
      "z": 0.6162
    },
    "heading": {
      "hx": -0.9994,
      "hy": 0.0353
    },
    "quaternion": {
      "w": 0.9998,
      "x": -0.0,
      "y": -0.0,
      "z": -0.0191
    },
    "bbox_2d": [
      {
        "x": 2.3413,
        "y": -7.2424
      },
      {
        "x": 2.319,
        "y": -7.8741
      },
      {
        "x": 1.601,
        "y": -7.8487
      },
      {
        "x": 1.6233,
        "y": -7.217
      }
    ],
    "z_range": [
      0.6162,
      3.1162
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_01_16_1",
    "name": "chair_01_16_platform_1",
    "base_object": "frl_apartment_chair_01_16",
    "centroid": {
      "x": 1.1583,
      "y": -7.5112,
      "z": 0.3877
    },
    "heading": {
      "hx": -1.0,
      "hy": -0.0029
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": 0.0,
      "z": -0.0
    },
    "bbox_2d": [
      {
        "x": 1.4082,
        "y": -7.2359
      },
      {
        "x": 1.4098,
        "y": -7.7851
      },
      {
        "x": 0.9083,
        "y": -7.7866
      },
      {
        "x": 0.9067,
        "y": -7.2374
      }
    ],
    "z_range": [
      0.3877,
      2.8877
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_01_16_2",
    "name": "chair_01_16_platform_2",
    "base_object": "frl_apartment_chair_01_16",
    "centroid": {
      "x": 1.158,
      "y": -7.5902,
      "z": 0.6162
    },
    "heading": {
      "hx": -1.0,
      "hy": -0.0029
    },
    "quaternion": {
      "w": 1.0,
      "x": 0.0,
      "y": 0.0,
      "z": -0.0
    },
    "bbox_2d": [
      {
        "x": 1.5163,
        "y": -7.2731
      },
      {
        "x": 1.5182,
        "y": -7.9052
      },
      {
        "x": 0.7997,
        "y": -7.9073
      },
      {
        "x": 0.7979,
        "y": -7.2752
      }
    ],
    "z_range": [
      0.6162,
      3.1162
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_0",
    "name": "chestOfDrawers_01_3_body_platform_0",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.5324,
      "z": 0.1246
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 1.3302
      },
      {
        "x": -2.4306,
        "y": 1.3301
      },
      {
        "x": -2.4306,
        "y": 1.7346
      },
      {
        "x": -2.1553,
        "y": 1.7347
      }
    ],
    "z_range": [
      0.1246,
      0.6024
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_1",
    "name": "chestOfDrawers_01_3_body_platform_1",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.1185,
      "z": 0.127
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 0.9195
      },
      {
        "x": -2.4305,
        "y": 0.9194
      },
      {
        "x": -2.4306,
        "y": 1.3175
      },
      {
        "x": -2.1552,
        "y": 1.3176
      }
    ],
    "z_range": [
      0.127,
      0.6048
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_2",
    "name": "chestOfDrawers_01_3_body_platform_2",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.5354,
      "z": 0.6521
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 1.3332
      },
      {
        "x": -2.4306,
        "y": 1.3332
      },
      {
        "x": -2.4306,
        "y": 1.7377
      },
      {
        "x": -2.1553,
        "y": 1.7377
      }
    ],
    "z_range": [
      0.6521,
      1.065
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_3",
    "name": "chestOfDrawers_01_3_body_platform_3",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.1216,
      "z": 0.6545
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 0.9225
      },
      {
        "x": -2.4305,
        "y": 0.9225
      },
      {
        "x": -2.4306,
        "y": 1.3206
      },
      {
        "x": -2.1552,
        "y": 1.3206
      }
    ],
    "z_range": [
      0.6545,
      1.0674
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_4",
    "name": "chestOfDrawers_01_3_body_platform_4",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.5379,
      "z": 1.0776
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 1.3357
      },
      {
        "x": -2.4306,
        "y": 1.3356
      },
      {
        "x": -2.4306,
        "y": 1.7401
      },
      {
        "x": -2.1553,
        "y": 1.7402
      }
    ],
    "z_range": [
      1.0776,
      1.2373
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_5",
    "name": "chestOfDrawers_01_3_body_platform_5",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.2929,
      "y": 1.124,
      "z": 1.08
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1552,
        "y": 0.925
      },
      {
        "x": -2.4305,
        "y": 0.925
      },
      {
        "x": -2.4306,
        "y": 1.323
      },
      {
        "x": -2.1552,
        "y": 1.3231
      }
    ],
    "z_range": [
      1.08,
      1.2397
    ],
    "children_objects": []
  },
  {
    "platform_id": "chestOfDrawers_01_3_body_6",
    "name": "chestOfDrawers_01_3_body_platform_6",
    "base_object": "chestOfDrawers_01_3_body",
    "centroid": {
      "x": -2.3186,
      "y": 1.3337,
      "z": 1.2796
    },
    "heading": {
      "hx": -0.0001,
      "hy": 1.0
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0029,
      "y": -0.0,
      "z": 0.0
    },
    "bbox_2d": [
      {
        "x": -2.1326,
        "y": 0.8925
      },
      {
        "x": -2.5045,
        "y": 0.8925
      },
      {
        "x": -2.5046,
        "y": 1.775
      },
      {
        "x": -2.1328,
        "y": 1.775
      }
    ],
    "z_range": [
      1.2796,
      3.7796
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_chair_04_47_0",
    "name": "chair_04_47_platform_0",
    "base_object": "frl_apartment_chair_04_47",
    "centroid": {
      "x": -2.1587,
      "y": 2.1884,
      "z": 0.529
    },
    "heading": {
      "hx": 0.2256,
      "hy": 0.9742
    },
    "quaternion": {
      "w": 0.6328,
      "x": 0.342,
      "y": 0.202,
      "z": -0.6647
    },
    "bbox_2d": [
      {
        "x": -2.0151,
        "y": 2.0886
      },
      {
        "x": -2.3315,
        "y": 2.1619
      },
      {
        "x": -2.3023,
        "y": 2.2882
      },
      {
        "x": -1.9859,
        "y": 2.215
      }
    ],
    "z_range": [
      0.529,
      3.029
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_table_02_40_0",
    "name": "table_02_40_platform_0",
    "base_object": "frl_apartment_table_02_40",
    "centroid": {
      "x": 0.7211,
      "y": -2.4649,
      "z": 0.2215
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": {
      "w": 0.3068,
      "x": -0.0,
      "y": -0.0,
      "z": 0.9518
    },
    "bbox_2d": [
      {
        "x": 0.3224,
        "y": -2.393
      },
      {
        "x": 0.5264,
        "y": -2.1096
      },
      {
        "x": 1.1197,
        "y": -2.5368
      },
      {
        "x": 0.9157,
        "y": -2.8202
      }
    ],
    "z_range": [
      0.2215,
      0.4972
    ],
    "children_objects": []
  },
  {
    "platform_id": "frl_apartment_table_02_40_1",
    "name": "table_02_40_platform_1",
    "base_object": "frl_apartment_table_02_40",
    "centroid": {
      "x": 0.6709,
      "y": -2.2102,
      "z": 0.2215
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": {
      "w": 0.3068,
      "x": -0.0,
      "y": -0.0,
      "z": 0.9518
    },
    "bbox_2d": [
      {
        "x": 0.221,
        "y": -2.3153
      },
      {
        "x": 0.6279,
        "y": -1.7502
      },
      {
        "x": 1.1208,
        "y": -2.1051
      },
      {
        "x": 0.7139,
        "y": -2.6702
      }
    ],
    "z_range": [
      0.2215,
      0.4972
    ],
    "children_objects": [
      "kitchenware_dark_blue_saucepan_on_wooden_shelf_43"
    ]
  },
  {
    "platform_id": "frl_apartment_table_02_40_2",
    "name": "table_02_40_platform_2",
    "base_object": "frl_apartment_table_02_40",
    "centroid": {
      "x": 1.0722,
      "y": -2.6911,
      "z": 0.2215
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": {
      "w": 0.3068,
      "x": -0.0,
      "y": -0.0,
      "z": 0.9518
    },
    "bbox_2d": [
      {
        "x": 0.7107,
        "y": -2.6748
      },
      {
        "x": 0.942,
        "y": -2.3534
      },
      {
        "x": 1.4337,
        "y": -2.7074
      },
      {
        "x": 1.2023,
        "y": -3.0288
      }
    ],
    "z_range": [
      0.2215,
      0.4972
    ],
    "children_objects": [
      "kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
      "kitchenware_gray_cast_iron_casserole_on_wooden_platform_44"
    ]
  },
  {
    "platform_id": "frl_apartment_table_02_40_3",
    "name": "table_02_40_platform_3",
    "base_object": "frl_apartment_table_02_40",
    "centroid": {
      "x": 0.8275,
      "y": -2.5112,
      "z": 0.5535
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": {
      "w": 0.3068,
      "x": -0.0,
      "y": -0.0,
      "z": 0.9518
    },
    "bbox_2d": [
      {
        "x": 0.2177,
        "y": -2.3199
      },
      {
        "x": 0.4526,
        "y": -1.9936
      },
      {
        "x": 1.4372,
        "y": -2.7026
      },
      {
        "x": 1.2023,
        "y": -3.0288
      }
    ],
    "z_range": [
      0.5535,
      0.8802
    ],
    "children_objects": [
      "furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
      "kitchenware_gray_shallow_bowl_on_wooden_shelf_41"
    ]
  },
  {
    "platform_id": "frl_apartment_table_02_40_4",
    "name": "table_02_40_platform_4",
    "base_object": "frl_apartment_table_02_40",
    "centroid": {
      "x": 0.921,
      "y": -2.3779,
      "z": 0.9288
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": {
      "w": 0.3068,
      "x": -0.0,
      "y": -0.0,
      "z": 0.9518
    },
    "bbox_2d": [
      {
        "x": 0.1652,
        "y": -2.3312
      },
      {
        "x": 0.637,
        "y": -1.6759
      },
      {
        "x": 1.6769,
        "y": -2.4246
      },
      {
        "x": 1.2051,
        "y": -3.0799
      }
    ],
    "z_range": [
      0.9288,
      3.4288
    ],
    "children_objects": [
      "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
      "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82"
    ]
  },
  {
    "platform_id": "frl_apartment_tvstand_89_0",
    "name": "tvstand_89_platform_0",
    "base_object": "frl_apartment_tvstand_89",
    "centroid": {
      "x": 3.152,
      "y": -7.6818,
      "z": 0.6121
    },
    "heading": {
      "hx": -1.0,
      "hy": 0.0009
    },
    "quaternion": {
      "w": 1.0,
      "x": -0.0,
      "y": -0.0,
      "z": -0.0
    },
    "bbox_2d": [
      {
        "x": 3.9102,
        "y": -7.4309
      },
      {
        "x": 3.9098,
        "y": -7.934
      },
      {
        "x": 2.3938,
        "y": -7.9326
      },
      {
        "x": 2.3942,
        "y": -7.4296
      }
    ],
    "z_range": [
      0.6121,
      3.1121
    ],
    "children_objects": [
      "electronics_black_wireless_mouse_on_blue_table_platform_20",
      "electronics_black_wireless_mouse_on_blue_table_platform_19"
    ]
  },
  {
    "platform_id": "cabinet_4_body_0",
    "name": "cabinet_4_body_platform_0",
    "base_object": "cabinet_4_body",
    "centroid": {
      "x": 1.3488,
      "y": -1.9,
      "z": 0.223
    },
    "heading": {
      "hx": -0.8024,
      "hy": 0.5968
    },
    "quaternion": {
      "w": 0.3153,
      "x": 0.0,
      "y": 0.0,
      "z": 0.949
    },
    "bbox_2d": [
      {
        "x": 2.1002,
        "y": -2.2016
      },
      {
        "x": 1.8539,
        "y": -2.5328
      },
      {
        "x": 0.5973,
        "y": -1.5983
      },
      {
        "x": 0.8436,
        "y": -1.2671
      }
    ],
    "z_range": [
      0.223,
      0.7739
    ],
    "children_objects": []
  },
  {
    "platform_id": "cabinet_4_body_1",
    "name": "cabinet_4_body_platform_1",
    "base_object": "cabinet_4_body",
    "centroid": {
      "x": 1.3455,
      "y": -1.9042,
      "z": 0.7928
    },
    "heading": {
      "hx": -0.8024,
      "hy": 0.5968
    },
    "quaternion": {
      "w": 0.3153,
      "x": 0.0,
      "y": 0.0,
      "z": 0.949
    },
    "bbox_2d": [
      {
        "x": 2.1088,
        "y": -2.208
      },
      {
        "x": 1.8561,
        "y": -2.5477
      },
      {
        "x": 0.5823,
        "y": -1.6003
      },
      {
        "x": 0.8349,
        "y": -1.2606
      }
    ],
    "z_range": [
      0.7928,
      3.2928
    ],
    "children_objects": []
  }
],
  "objects": [
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_11",
    "name": "furniture_light_blue_square_cushion_on_sofa_platform_11",
    "category": "furniture",
    "centroid": {
      "x": 3.7302,
      "y": -5.2892,
      "z": 0.4997
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 3.9576,
        "y": -5.5014
      },
      {
        "x": 3.5195,
        "y": -5.5011
      },
      {
        "x": 3.5198,
        "y": -5.0785
      },
      {
        "x": 3.9579,
        "y": -5.0788
      }
    ],
    "z_range": [
      0.3966,
      0.6206
    ],
    "size": {
      "x_length": 0.4384,
      "y_length": 0.4229,
      "z_length": 0.224
    },
    "on_platform": "frl_apartment_sofa_10_1"
  },
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_12",
    "name": "furniture_light_blue_square_cushion_on_sofa_platform_12",
    "category": "furniture",
    "centroid": {
      "x": 3.7706,
      "y": -5.8232,
      "z": 0.5136
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 3.9756,
        "y": -6.0441
      },
      {
        "x": 3.5466,
        "y": -6.0438
      },
      {
        "x": 3.5469,
        "y": -5.6072
      },
      {
        "x": 3.9759,
        "y": -5.6075
      }
    ],
    "z_range": [
      0.4012,
      0.6284
    ],
    "size": {
      "x_length": 0.4294,
      "y_length": 0.4369,
      "z_length": 0.2272
    },
    "on_platform": "frl_apartment_sofa_10_1"
  },
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_9",
    "name": "furniture_light_blue_square_cushion_on_sofa_platform_9",
    "category": "furniture",
    "centroid": {
      "x": 3.7844,
      "y": -4.7702,
      "z": 0.5136
    },
    "heading": {
      "hx": 0.0007,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.0132,
        "y": -5.0316
      },
      {
        "x": 3.5527,
        "y": -5.0312
      },
      {
        "x": 3.553,
        "y": -4.5328
      },
      {
        "x": 4.0135,
        "y": -4.5331
      }
    ],
    "z_range": [
      0.4012,
      0.6283
    ],
    "size": {
      "x_length": 0.4609,
      "y_length": 0.4988,
      "z_length": 0.2271
    },
    "on_platform": "frl_apartment_sofa_10_2"
  },
  {
    "object_id": "furniture_gray_tissue_box_on_platform_17",
    "name": "furniture_gray_tissue_box_on_platform_17",
    "category": "furniture",
    "centroid": {
      "x": 0.5355,
      "y": -7.703,
      "z": 0.766
    },
    "heading": {
      "hx": -1.0,
      "hy": -0.0003
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.7086,
        "y": -7.4608
      },
      {
        "x": 0.7087,
        "y": -7.9424
      },
      {
        "x": 0.366,
        "y": -7.9425
      },
      {
        "x": 0.3658,
        "y": -7.4608
      }
    ],
    "z_range": [
      0.4218,
      1.0439
    ],
    "size": {
      "x_length": 0.3429,
      "y_length": 0.4818,
      "z_length": 0.6221
    },
    "on_platform": "frl_apartment_stool_02_18_0"
  },
  {
    "object_id": "footwear_red_high_heel_on_shelf_platform_79",
    "name": "footwear_red_high_heel_on_shelf_platform_79",
    "category": "footwear",
    "centroid": {
      "x": -2.2628,
      "y": 2.667,
      "z": 0.3487
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1393,
        "y": 2.7854
      },
      {
        "x": -2.18,
        "y": 2.5139
      },
      {
        "x": -2.3903,
        "y": 2.5454
      },
      {
        "x": -2.3495,
        "y": 2.8169
      }
    ],
    "z_range": [
      0.3039,
      0.4157
    ],
    "size": {
      "x_length": 0.251,
      "y_length": 0.3031,
      "z_length": 0.1118
    },
    "on_platform": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "footwear_black_sneaker_on_platform_77",
    "name": "footwear_black_sneaker_on_platform_77",
    "category": "footwear",
    "centroid": {
      "x": -1.732,
      "y": 2.6097,
      "z": 0.351
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.5586,
        "y": 2.7446
      },
      {
        "x": -1.6113,
        "y": 2.3933
      },
      {
        "x": -1.895,
        "y": 2.4358
      },
      {
        "x": -1.8423,
        "y": 2.7871
      }
    ],
    "z_range": [
      0.3042,
      0.4308
    ],
    "size": {
      "x_length": 0.3364,
      "y_length": 0.3938,
      "z_length": 0.1266
    },
    "on_platform": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "footwear_brown_leather_dress_shoe_on_platform_78",
    "name": "footwear_brown_leather_dress_shoe_on_platform_78",
    "category": "footwear",
    "centroid": {
      "x": -2.0251,
      "y": 2.6324,
      "z": 0.3509
    },
    "heading": {
      "hx": -0.9889,
      "hy": 0.1483
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.8349,
        "y": 2.7671
      },
      {
        "x": -1.888,
        "y": 2.4127
      },
      {
        "x": -2.2007,
        "y": 2.4596
      },
      {
        "x": -2.1476,
        "y": 2.814
      }
    ],
    "z_range": [
      0.3044,
      0.4308
    ],
    "size": {
      "x_length": 0.3659,
      "y_length": 0.4014,
      "z_length": 0.1264
    },
    "on_platform": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "electronics_black_wall_bracket_on_platform_2",
    "name": "electronics_black_wall_bracket_on_platform_2",
    "category": "electronics",
    "centroid": {
      "x": -2.2426,
      "y": -0.1335,
      "z": 0.2261
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.9468,
        "y": -0.5626
      },
      {
        "x": -2.5052,
        "y": -0.5682
      },
      {
        "x": -2.5127,
        "y": 0.1928
      },
      {
        "x": -1.9544,
        "y": 0.1984
      }
    ],
    "z_range": [
      0.0804,
      0.4042
    ],
    "size": {
      "x_length": 0.5659,
      "y_length": 0.7665,
      "z_length": 0.3238
    },
    "on_platform": "kitchen_counter_1_body_0"
  },
  {
    "object_id": "kitchenware_white_cylindrical_spice_jars_on_platform_59",
    "name": "kitchenware_white_cylindrical_spice_jars_on_platform_59",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3934,
      "y": -0.7398,
      "z": 0.9136
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.2806,
        "y": -0.8975
      },
      {
        "x": -2.482,
        "y": -0.8995
      },
      {
        "x": -2.4851,
        "y": -0.5821
      },
      {
        "x": -2.2837,
        "y": -0.58
      }
    ],
    "z_range": [
      0.8665,
      0.9697
    ],
    "size": {
      "x_length": 0.2045,
      "y_length": 0.3195,
      "z_length": 0.1032
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_66",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_66",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3754,
      "y": -0.7068,
      "z": 0.9633
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3492,
        "y": -0.7336
      },
      {
        "x": -2.401,
        "y": -0.7341
      },
      {
        "x": -2.4015,
        "y": -0.68
      },
      {
        "x": -2.3497,
        "y": -0.6795
      }
    ],
    "z_range": [
      0.9123,
      1.0196
    ],
    "size": {
      "x_length": 0.0524,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_67",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_67",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3751,
      "y": -0.6513,
      "z": 0.9633
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3491,
        "y": -0.678
      },
      {
        "x": -2.4007,
        "y": -0.6786
      },
      {
        "x": -2.4012,
        "y": -0.6245
      },
      {
        "x": -2.3496,
        "y": -0.6239
      }
    ],
    "z_range": [
      0.9123,
      1.0196
    ],
    "size": {
      "x_length": 0.0522,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_65",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_65",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3752,
      "y": -0.7615,
      "z": 0.9633
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.349,
        "y": -0.7883
      },
      {
        "x": -2.4009,
        "y": -0.7888
      },
      {
        "x": -2.4014,
        "y": -0.7347
      },
      {
        "x": -2.3496,
        "y": -0.7342
      }
    ],
    "z_range": [
      0.9123,
      1.0196
    ],
    "size": {
      "x_length": 0.0524,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_64",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_64",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3756,
      "y": -0.8186,
      "z": 0.9633
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3495,
        "y": -0.8454
      },
      {
        "x": -2.4011,
        "y": -0.8459
      },
      {
        "x": -2.4016,
        "y": -0.7918
      },
      {
        "x": -2.35,
        "y": -0.7913
      }
    ],
    "z_range": [
      0.9123,
      1.0196
    ],
    "size": {
      "x_length": 0.0522,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_62",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_62",
    "category": "kitchenware",
    "centroid": {
      "x": -2.4441,
      "y": -0.7154,
      "z": 0.9913
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.418,
        "y": -0.7422
      },
      {
        "x": -2.4697,
        "y": -0.7427
      },
      {
        "x": -2.4702,
        "y": -0.6886
      },
      {
        "x": -2.4186,
        "y": -0.6881
      }
    ],
    "z_range": [
      0.9403,
      1.0476
    ],
    "size": {
      "x_length": 0.0522,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_60",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_60",
    "category": "kitchenware",
    "centroid": {
      "x": -2.4442,
      "y": -0.8247,
      "z": 0.9913
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.4181,
        "y": -0.8515
      },
      {
        "x": -2.4697,
        "y": -0.852
      },
      {
        "x": -2.4703,
        "y": -0.7979
      },
      {
        "x": -2.4186,
        "y": -0.7974
      }
    ],
    "z_range": [
      0.9403,
      1.0476
    ],
    "size": {
      "x_length": 0.0522,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_61",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_61",
    "category": "kitchenware",
    "centroid": {
      "x": -2.4436,
      "y": -0.77,
      "z": 0.9913
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.4176,
        "y": -0.7968
      },
      {
        "x": -2.4691,
        "y": -0.7973
      },
      {
        "x": -2.4697,
        "y": -0.7432
      },
      {
        "x": -2.4181,
        "y": -0.7427
      }
    ],
    "z_range": [
      0.9403,
      1.0476
    ],
    "size": {
      "x_length": 0.0521,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_63",
    "name": "kitchenware_orange_spice_shaker_on_white_platform_63",
    "category": "kitchenware",
    "centroid": {
      "x": -2.4395,
      "y": -0.6614,
      "z": 0.9913
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.4134,
        "y": -0.6882
      },
      {
        "x": -2.465,
        "y": -0.6887
      },
      {
        "x": -2.4656,
        "y": -0.6346
      },
      {
        "x": -2.414,
        "y": -0.6341
      }
    ],
    "z_range": [
      0.9403,
      1.0476
    ],
    "size": {
      "x_length": 0.0521,
      "y_length": 0.0546,
      "z_length": 0.1073
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
    "name": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
    "category": "kitchenware",
    "centroid": {
      "x": -2.4622,
      "y": -0.3173,
      "z": 0.9944
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.4005,
        "y": -0.4943
      },
      {
        "x": -2.5205,
        "y": -0.4955
      },
      {
        "x": -2.524,
        "y": -0.1403
      },
      {
        "x": -2.404,
        "y": -0.1391
      }
    ],
    "z_range": [
      0.8666,
      1.1223
    ],
    "size": {
      "x_length": 0.1235,
      "y_length": 0.3564,
      "z_length": 0.2557
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "decor_photo_frame_with_dog_picture_on_platform_70",
    "name": "decor_photo_frame_with_dog_picture_on_platform_70",
    "category": "decor",
    "centroid": {
      "x": -2.2304,
      "y": -0.4031,
      "z": 0.97
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.0811,
        "y": -0.4856
      },
      {
        "x": -2.3769,
        "y": -0.4886
      },
      {
        "x": -2.3785,
        "y": -0.3261
      },
      {
        "x": -2.0827,
        "y": -0.3231
      }
    ],
    "z_range": [
      0.8666,
      1.0758
    ],
    "size": {
      "x_length": 0.2975,
      "y_length": 0.1655,
      "z_length": 0.2092
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_octagonal_plate_on_black_platform_50",
    "name": "kitchenware_white_octagonal_plate_on_black_platform_50",
    "category": "kitchenware",
    "centroid": {
      "x": -2.0298,
      "y": -2.6329,
      "z": 0.8822
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.8991,
        "y": -2.7613
      },
      {
        "x": -2.1579,
        "y": -2.7639
      },
      {
        "x": -2.1604,
        "y": -2.5045
      },
      {
        "x": -1.9017,
        "y": -2.502
      }
    ],
    "z_range": [
      0.8675,
      0.895
    ],
    "size": {
      "x_length": 0.2613,
      "y_length": 0.2619,
      "z_length": 0.0275
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_beige_coffee_cup_on_black_table_platform_51",
    "name": "kitchenware_beige_coffee_cup_on_black_table_platform_51",
    "category": "kitchenware",
    "centroid": {
      "x": -2.0333,
      "y": -2.6401,
      "z": 0.9167
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.9648,
        "y": -2.7319
      },
      {
        "x": -2.1034,
        "y": -2.7333
      },
      {
        "x": -2.105,
        "y": -2.5711
      },
      {
        "x": -1.9664,
        "y": -2.5698
      }
    ],
    "z_range": [
      0.882,
      0.9575
    ],
    "size": {
      "x_length": 0.1402,
      "y_length": 0.1635,
      "z_length": 0.0755
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_flower_patterned_cream_jug_on_platform_55",
    "name": "kitchenware_white_flower_patterned_cream_jug_on_platform_55",
    "category": "kitchenware",
    "centroid": {
      "x": -2.2145,
      "y": -2.0379,
      "z": 0.9138
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1613,
        "y": -2.0929
      },
      {
        "x": -2.2671,
        "y": -2.0939
      },
      {
        "x": -2.2684,
        "y": -1.9638
      },
      {
        "x": -2.1626,
        "y": -1.9628
      }
    ],
    "z_range": [
      0.8676,
      0.9691
    ],
    "size": {
      "x_length": 0.1071,
      "y_length": 0.1312,
      "z_length": 0.1015
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_shallow_bowl_on_table_platform_57",
    "name": "kitchenware_white_shallow_bowl_on_table_platform_57",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3942,
      "y": -1.9151,
      "z": 0.9412
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3478,
        "y": -1.9628
      },
      {
        "x": -2.4396,
        "y": -1.9637
      },
      {
        "x": -2.4406,
        "y": -1.8674
      },
      {
        "x": -2.3488,
        "y": -1.8665
      }
    ],
    "z_range": [
      0.8676,
      1.05
    ],
    "size": {
      "x_length": 0.0928,
      "y_length": 0.0972,
      "z_length": 0.1824
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_small_coffee_mug_on_black_table_platform_52",
    "name": "kitchenware_white_small_coffee_mug_on_black_table_platform_52",
    "category": "kitchenware",
    "centroid": {
      "x": -2.2409,
      "y": -2.5617,
      "z": 0.9017
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1857,
        "y": -2.6355
      },
      {
        "x": -2.293,
        "y": -2.6366
      },
      {
        "x": -2.2944,
        "y": -2.5044
      },
      {
        "x": -2.187,
        "y": -2.5033
      }
    ],
    "z_range": [
      0.8677,
      0.9467
    ],
    "size": {
      "x_length": 0.1087,
      "y_length": 0.1333,
      "z_length": 0.079
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_gray_cylindrical_container_on_platform_72",
    "name": "kitchenware_gray_cylindrical_container_on_platform_72",
    "category": "kitchenware",
    "centroid": {
      "x": -2.1923,
      "y": -0.309,
      "z": 0.9386
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1334,
        "y": -0.3665
      },
      {
        "x": -2.2506,
        "y": -0.3677
      },
      {
        "x": -2.2518,
        "y": -0.2504
      },
      {
        "x": -2.1345,
        "y": -0.2492
      }
    ],
    "z_range": [
      0.8678,
      1.018
    ],
    "size": {
      "x_length": 0.1185,
      "y_length": 0.1184,
      "z_length": 0.1502
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_blue_thermos_bottle_73",
    "name": "kitchenware_blue_thermos_bottle_73",
    "category": "kitchenware",
    "centroid": {
      "x": -2.2383,
      "y": -0.2029,
      "z": 0.961
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1884,
        "y": -0.2541
      },
      {
        "x": -2.2873,
        "y": -0.2551
      },
      {
        "x": -2.2883,
        "y": -0.1516
      },
      {
        "x": -2.1894,
        "y": -0.1506
      }
    ],
    "z_range": [
      0.8678,
      1.072
    ],
    "size": {
      "x_length": 0.0999,
      "y_length": 0.1045,
      "z_length": 0.2042
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_round_bowl_on_black_table_platform_54",
    "name": "kitchenware_white_round_bowl_on_black_table_platform_54",
    "category": "kitchenware",
    "centroid": {
      "x": -2.0213,
      "y": -1.9621,
      "z": 0.9038
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.9402,
        "y": -2.0415
      },
      {
        "x": -2.1007,
        "y": -2.0431
      },
      {
        "x": -2.1023,
        "y": -1.8826
      },
      {
        "x": -1.9418,
        "y": -1.881
      }
    ],
    "z_range": [
      0.8679,
      0.9423
    ],
    "size": {
      "x_length": 0.1621,
      "y_length": 0.1622,
      "z_length": 0.0744
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_transparent_bowl_on_platform_56",
    "name": "kitchenware_transparent_bowl_on_platform_56",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3765,
      "y": -2.0388,
      "z": 0.9079
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.2883,
        "y": -2.1292
      },
      {
        "x": -2.4578,
        "y": -2.1309
      },
      {
        "x": -2.4595,
        "y": -1.9615
      },
      {
        "x": -2.29,
        "y": -1.9598
      }
    ],
    "z_range": [
      0.8679,
      0.9815
    ],
    "size": {
      "x_length": 0.1712,
      "y_length": 0.1711,
      "z_length": 0.1136
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_wooden_knife_block_on_platform_71",
    "name": "kitchenware_wooden_knife_block_on_platform_71",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3513,
      "y": -0.3304,
      "z": 0.9946
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3019,
        "y": -0.4028
      },
      {
        "x": -2.3991,
        "y": -0.4038
      },
      {
        "x": -2.4006,
        "y": -0.2536
      },
      {
        "x": -2.3034,
        "y": -0.2526
      }
    ],
    "z_range": [
      0.868,
      1.1682
    ],
    "size": {
      "x_length": 0.0987,
      "y_length": 0.1512,
      "z_length": 0.3002
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "furniture_brown_cushion_on_platform_53",
    "name": "furniture_brown_cushion_on_platform_53",
    "category": "furniture",
    "centroid": {
      "x": -2.0809,
      "y": -2.255,
      "z": 0.9081
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.8915,
        "y": -2.4333
      },
      {
        "x": -2.2712,
        "y": -2.4371
      },
      {
        "x": -2.2748,
        "y": -2.0749
      },
      {
        "x": -1.8951,
        "y": -2.0711
      }
    ],
    "z_range": [
      0.8688,
      0.9534
    ],
    "size": {
      "x_length": 0.3833,
      "y_length": 0.366,
      "z_length": 0.0846
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
    "name": "kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3231,
      "y": -0.0002,
      "z": 1.0061
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.1604,
        "y": -0.0881
      },
      {
        "x": -2.4478,
        "y": -0.091
      },
      {
        "x": -2.4496,
        "y": 0.0892
      },
      {
        "x": -2.1622,
        "y": 0.092
      }
    ],
    "z_range": [
      0.869,
      1.167
    ],
    "size": {
      "x_length": 0.2892,
      "y_length": 0.183,
      "z_length": 0.298
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "container_brown_paper_food_box_on_table_platform_0",
    "name": "container_brown_paper_food_box_on_table_platform_0",
    "category": "container",
    "centroid": {
      "x": -1.9957,
      "y": -0.058,
      "z": 1.0017
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.8467,
        "y": -0.2208
      },
      {
        "x": -2.1413,
        "y": -0.2238
      },
      {
        "x": -2.1446,
        "y": 0.1049
      },
      {
        "x": -1.8499,
        "y": 0.1078
      }
    ],
    "z_range": [
      0.8693,
      1.1782
    ],
    "size": {
      "x_length": 0.2979,
      "y_length": 0.3316,
      "z_length": 0.3089
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_cylindrical_cup_on_black_platform_68",
    "name": "kitchenware_white_cylindrical_cup_on_black_platform_68",
    "category": "kitchenware",
    "centroid": {
      "x": -2.3675,
      "y": -0.5325,
      "z": 0.9242
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -2.3167,
        "y": -0.5815
      },
      {
        "x": -2.4174,
        "y": -0.5825
      },
      {
        "x": -2.4184,
        "y": -0.4831
      },
      {
        "x": -2.3177,
        "y": -0.4821
      }
    ],
    "z_range": [
      0.8694,
      0.9933
    ],
    "size": {
      "x_length": 0.1017,
      "y_length": 0.1004,
      "z_length": 0.1239
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_brown_black_coffee_grinder_on_table_platform_58",
    "name": "kitchenware_brown_black_coffee_grinder_on_table_platform_58",
    "category": "kitchenware",
    "centroid": {
      "x": -2.1578,
      "y": -1.132,
      "z": 1.0119
    },
    "heading": {
      "hx": -0.0099,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": -1.9986,
        "y": -1.2357
      },
      {
        "x": -2.375,
        "y": -1.2394
      },
      {
        "x": -2.3771,
        "y": -1.0265
      },
      {
        "x": -2.0008,
        "y": -1.0228
      }
    ],
    "z_range": [
      0.8696,
      1.2272
    ],
    "size": {
      "x_length": 0.3785,
      "y_length": 0.2166,
      "z_length": 0.3576
    },
    "on_platform": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "decor_brown_wooden_mantle_clock_on_wall_shelf_5",
    "name": "decor_brown_wooden_mantle_clock_on_wall_shelf_5",
    "category": "decor",
    "centroid": {
      "x": 4.1744,
      "y": -3.9458,
      "z": 0.5627
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.2091,
        "y": -4.0758
      },
      {
        "x": 4.1429,
        "y": -4.0775
      },
      {
        "x": 4.136,
        "y": -3.8164
      },
      {
        "x": 4.2023,
        "y": -3.8147
      }
    ],
    "z_range": [
      0.5148,
      0.6803
    ],
    "size": {
      "x_length": 0.0731,
      "y_length": 0.2628,
      "z_length": 0.1655
    },
    "on_platform": "frl_apartment_wall_cabinet_01_4_1"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_88",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_88",
    "category": "book",
    "centroid": {
      "x": 4.198,
      "y": -4.0144,
      "z": 1.384
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.2771,
        "y": -4.1024
      },
      {
        "x": 4.1258,
        "y": -4.1064
      },
      {
        "x": 4.121,
        "y": -3.9247
      },
      {
        "x": 4.2723,
        "y": -3.9208
      }
    ],
    "z_range": [
      1.3685,
      1.3995
    ],
    "size": {
      "x_length": 0.156,
      "y_length": 0.1856,
      "z_length": 0.0311
    },
    "on_platform": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_87",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_87",
    "category": "book",
    "centroid": {
      "x": 4.1979,
      "y": -4.0139,
      "z": 1.41
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.277,
        "y": -4.1019
      },
      {
        "x": 4.1257,
        "y": -4.1059
      },
      {
        "x": 4.1209,
        "y": -3.9242
      },
      {
        "x": 4.2722,
        "y": -3.9202
      }
    ],
    "z_range": [
      1.3945,
      1.4255
    ],
    "size": {
      "x_length": 0.1561,
      "y_length": 0.1857,
      "z_length": 0.031
    },
    "on_platform": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_86",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_86",
    "category": "book",
    "centroid": {
      "x": 4.1966,
      "y": -3.8617,
      "z": 1.411
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.2764,
        "y": -3.9492
      },
      {
        "x": 4.1237,
        "y": -3.9532
      },
      {
        "x": 4.119,
        "y": -3.7742
      },
      {
        "x": 4.2717,
        "y": -3.7702
      }
    ],
    "z_range": [
      1.3685,
      1.4535
    ],
    "size": {
      "x_length": 0.1574,
      "y_length": 0.1829,
      "z_length": 0.085
    },
    "on_platform": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "container_brown_wooden_box_on_white_shelf_84",
    "name": "container_brown_wooden_box_on_white_shelf_84",
    "category": "container",
    "centroid": {
      "x": 4.1808,
      "y": -3.9582,
      "z": 1.7346
    },
    "heading": {
      "hx": -0.0262,
      "hy": 0.9997
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 4.3178,
        "y": -4.127
      },
      {
        "x": 4.0475,
        "y": -4.134
      },
      {
        "x": 4.0383,
        "y": -3.7844
      },
      {
        "x": 4.3086,
        "y": -3.7773
      }
    ],
    "z_range": [
      1.6601,
      1.834
    ],
    "size": {
      "x_length": 0.2795,
      "y_length": 0.3567,
      "z_length": 0.1738
    },
    "on_platform": "frl_apartment_wall_cabinet_01_4_5"
  },
  {
    "object_id": "electronics_white_security_camera_on_table_platform_49",
    "name": "electronics_white_security_camera_on_table_platform_49",
    "category": "electronics",
    "centroid": {
      "x": 0.4284,
      "y": 0.5712,
      "z": 0.7948
    },
    "heading": {
      "hx": -1.0,
      "hy": 0.0006
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.5264,
        "y": 0.6463
      },
      {
        "x": 0.5263,
        "y": 0.501
      },
      {
        "x": 0.3219,
        "y": 0.5011
      },
      {
        "x": 0.322,
        "y": 0.6464
      }
    ],
    "z_range": [
      0.751,
      0.8841
    ],
    "size": {
      "x_length": 0.2044,
      "y_length": 0.1454,
      "z_length": 0.1332
    },
    "on_platform": "frl_apartment_table_01_48_0"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_36",
    "name": "book_black_thin_softcover_on_table_platform_36",
    "category": "book",
    "centroid": {
      "x": 0.4827,
      "y": -5.6852,
      "z": 0.1875
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.5349,
        "y": -5.7616
      },
      {
        "x": 0.4292,
        "y": -5.7614
      },
      {
        "x": 0.4295,
        "y": -5.6086
      },
      {
        "x": 0.5352,
        "y": -5.6088
      }
    ],
    "z_range": [
      0.1798,
      0.1951
    ],
    "size": {
      "x_length": 0.1059,
      "y_length": 0.153,
      "z_length": 0.0153
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_37",
    "name": "book_black_thin_softcover_on_table_platform_37",
    "category": "book",
    "centroid": {
      "x": 0.5302,
      "y": -5.5127,
      "z": 0.1875
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6224,
        "y": -5.5981
      },
      {
        "x": 0.4383,
        "y": -5.5978
      },
      {
        "x": 0.4386,
        "y": -5.4272
      },
      {
        "x": 0.6227,
        "y": -5.4276
      }
    ],
    "z_range": [
      0.1799,
      0.1951
    ],
    "size": {
      "x_length": 0.1844,
      "y_length": 0.1708,
      "z_length": 0.0153
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_38",
    "name": "kitchenware_brown_handle_knife_on_white_table_platform_38",
    "category": "kitchenware",
    "centroid": {
      "x": 0.5526,
      "y": -5.5675,
      "z": 0.2033
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6064,
        "y": -5.6502
      },
      {
        "x": 0.4974,
        "y": -5.65
      },
      {
        "x": 0.4976,
        "y": -5.4849
      },
      {
        "x": 0.6067,
        "y": -5.4851
      }
    ],
    "z_range": [
      0.1951,
      0.2115
    ],
    "size": {
      "x_length": 0.1093,
      "y_length": 0.1653,
      "z_length": 0.0163
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_35",
    "name": "book_blue_thick_hardcover_on_white_shelf_platform_35",
    "category": "book",
    "centroid": {
      "x": 0.5781,
      "y": -5.0863,
      "z": 0.3117
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6715,
        "y": -5.126
      },
      {
        "x": 0.4781,
        "y": -5.1257
      },
      {
        "x": 0.4782,
        "y": -5.0462
      },
      {
        "x": 0.6717,
        "y": -5.0465
      }
    ],
    "z_range": [
      0.1799,
      0.4462
    ],
    "size": {
      "x_length": 0.1936,
      "y_length": 0.0798,
      "z_length": 0.2663
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_33",
    "name": "book_white_thick_hardcover_on_table_platform_33",
    "category": "book",
    "centroid": {
      "x": 0.4402,
      "y": -5.5789,
      "z": 0.5428
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.5278,
        "y": -5.6638
      },
      {
        "x": 0.3531,
        "y": -5.6635
      },
      {
        "x": 0.3533,
        "y": -5.4954
      },
      {
        "x": 0.5281,
        "y": -5.4957
      }
    ],
    "z_range": [
      0.5316,
      0.5541
    ],
    "size": {
      "x_length": 0.1751,
      "y_length": 0.1684,
      "z_length": 0.0225
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_32",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_32",
    "category": "book",
    "centroid": {
      "x": 0.5444,
      "y": -5.3026,
      "z": 0.5786
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6471,
        "y": -5.4117
      },
      {
        "x": 0.4386,
        "y": -5.4114
      },
      {
        "x": 0.439,
        "y": -5.192
      },
      {
        "x": 0.6475,
        "y": -5.1924
      }
    ],
    "z_range": [
      0.5316,
      0.6256
    ],
    "size": {
      "x_length": 0.2089,
      "y_length": 0.2197,
      "z_length": 0.094
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "book_navy_thick_hardcover_on_platform_34",
    "name": "book_navy_thick_hardcover_on_platform_34",
    "category": "book",
    "centroid": {
      "x": 0.5301,
      "y": -5.1551,
      "z": 0.5596
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6579,
        "y": -5.2707
      },
      {
        "x": 0.4031,
        "y": -5.2703
      },
      {
        "x": 0.4035,
        "y": -5.0366
      },
      {
        "x": 0.6583,
        "y": -5.0371
      }
    ],
    "z_range": [
      0.5316,
      0.5874
    ],
    "size": {
      "x_length": 0.2552,
      "y_length": 0.2341,
      "z_length": 0.0558
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "container_brown_wooden_box_on_white_shelf_85",
    "name": "container_brown_wooden_box_on_white_shelf_85",
    "category": "container",
    "centroid": {
      "x": 0.5361,
      "y": -5.6479,
      "z": 0.887
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.7079,
        "y": -5.7815
      },
      {
        "x": 0.3654,
        "y": -5.7809
      },
      {
        "x": 0.3658,
        "y": -5.5148
      },
      {
        "x": 0.7083,
        "y": -5.5154
      }
    ],
    "z_range": [
      0.8163,
      0.9843
    ],
    "size": {
      "x_length": 0.3429,
      "y_length": 0.2666,
      "z_length": 0.1681
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_2"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_25",
    "name": "book_black_thin_softcover_on_table_platform_25",
    "category": "book",
    "centroid": {
      "x": 0.4921,
      "y": -5.7272,
      "z": 1.1245
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.5454,
        "y": -5.8029
      },
      {
        "x": 0.4376,
        "y": -5.8028
      },
      {
        "x": 0.4379,
        "y": -5.6499
      },
      {
        "x": 0.5457,
        "y": -5.6501
      }
    ],
    "z_range": [
      1.1169,
      1.1322
    ],
    "size": {
      "x_length": 0.1081,
      "y_length": 0.153,
      "z_length": 0.0153
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_39",
    "name": "kitchenware_brown_handle_knife_on_white_table_platform_39",
    "category": "kitchenware",
    "centroid": {
      "x": 0.4935,
      "y": -5.6756,
      "z": 1.1403
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.549,
        "y": -5.7577
      },
      {
        "x": 0.4366,
        "y": -5.7575
      },
      {
        "x": 0.4369,
        "y": -5.5918
      },
      {
        "x": 0.5493,
        "y": -5.592
      }
    ],
    "z_range": [
      1.1321,
      1.1484
    ],
    "size": {
      "x_length": 0.1127,
      "y_length": 0.1659,
      "z_length": 0.0163
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "decor_abstract_colorful_painting_on_white_shelf_platform_22",
    "name": "decor_abstract_colorful_painting_on_white_shelf_platform_22",
    "category": "decor",
    "centroid": {
      "x": 0.5985,
      "y": -5.3904,
      "z": 1.2233
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6202,
        "y": -5.4746
      },
      {
        "x": 0.5842,
        "y": -5.4745
      },
      {
        "x": 0.5845,
        "y": -5.3061
      },
      {
        "x": 0.6205,
        "y": -5.3061
      }
    ],
    "z_range": [
      1.1169,
      1.3297
    ],
    "size": {
      "x_length": 0.0363,
      "y_length": 0.1685,
      "z_length": 0.2128
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_24",
    "name": "kitchenware_brown_handle_knife_on_white_table_platform_24",
    "category": "kitchenware",
    "centroid": {
      "x": 0.5515,
      "y": -5.1381,
      "z": 1.1986
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6067,
        "y": -5.1472
      },
      {
        "x": 0.4952,
        "y": -5.147
      },
      {
        "x": 0.4952,
        "y": -5.1291
      },
      {
        "x": 0.6067,
        "y": -5.1293
      }
    ],
    "z_range": [
      1.1169,
      1.282
    ],
    "size": {
      "x_length": 0.1115,
      "y_length": 0.018,
      "z_length": 0.1651
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_navy_thick_hardcover_on_platform_23",
    "name": "book_navy_thick_hardcover_on_platform_23",
    "category": "book",
    "centroid": {
      "x": 0.527,
      "y": -5.2174,
      "z": 1.2256
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.5935,
        "y": -5.2459
      },
      {
        "x": 0.4557,
        "y": -5.2457
      },
      {
        "x": 0.4558,
        "y": -5.1891
      },
      {
        "x": 0.5936,
        "y": -5.1894
      }
    ],
    "z_range": [
      1.1169,
      1.3364
    ],
    "size": {
      "x_length": 0.1379,
      "y_length": 0.0568,
      "z_length": 0.2195
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_29",
    "name": "book_white_thick_hardcover_on_table_platform_29",
    "category": "book",
    "centroid": {
      "x": 0.5189,
      "y": -5.1684,
      "z": 1.2034
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6029,
        "y": -5.1804
      },
      {
        "x": 0.4336,
        "y": -5.1802
      },
      {
        "x": 0.4336,
        "y": -5.1565
      },
      {
        "x": 0.6029,
        "y": -5.1568
      }
    ],
    "z_range": [
      1.1169,
      1.2917
    ],
    "size": {
      "x_length": 0.1693,
      "y_length": 0.024,
      "z_length": 0.1748
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_31",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_31",
    "category": "book",
    "centroid": {
      "x": 0.5457,
      "y": -5.1053,
      "z": 1.2057
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6192,
        "y": -5.1189
      },
      {
        "x": 0.47,
        "y": -5.1186
      },
      {
        "x": 0.47,
        "y": -5.0916
      },
      {
        "x": 0.6192,
        "y": -5.0919
      }
    ],
    "z_range": [
      1.1169,
      1.2963
    ],
    "size": {
      "x_length": 0.1493,
      "y_length": 0.0272,
      "z_length": 0.1794
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_28",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_28",
    "category": "book",
    "centroid": {
      "x": 0.5455,
      "y": -5.0705,
      "z": 1.2057
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.619,
        "y": -5.0846
      },
      {
        "x": 0.4697,
        "y": -5.0843
      },
      {
        "x": 0.4697,
        "y": -5.0563
      },
      {
        "x": 0.619,
        "y": -5.0566
      }
    ],
    "z_range": [
      1.1169,
      1.2963
    ],
    "size": {
      "x_length": 0.1494,
      "y_length": 0.0282,
      "z_length": 0.1794
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_30",
    "name": "book_maroon_thin_hardcover_on_white_shelf_platform_30",
    "category": "book",
    "centroid": {
      "x": 0.5441,
      "y": -5.0413,
      "z": 1.2057
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6176,
        "y": -5.0552
      },
      {
        "x": 0.4683,
        "y": -5.055
      },
      {
        "x": 0.4683,
        "y": -5.0273
      },
      {
        "x": 0.6176,
        "y": -5.0275
      }
    ],
    "z_range": [
      1.1169,
      1.2963
    ],
    "size": {
      "x_length": 0.1493,
      "y_length": 0.0279,
      "z_length": 0.1794
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_27",
    "name": "book_white_thick_hardcover_on_table_platform_27",
    "category": "book",
    "centroid": {
      "x": 0.5778,
      "y": -5.5999,
      "z": 1.4484
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6611,
        "y": -5.6813
      },
      {
        "x": 0.4928,
        "y": -5.6811
      },
      {
        "x": 0.4931,
        "y": -5.5184
      },
      {
        "x": 0.6614,
        "y": -5.5187
      }
    ],
    "z_range": [
      1.3984,
      1.4984
    ],
    "size": {
      "x_length": 0.1685,
      "y_length": 0.1629,
      "z_length": 0.1
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_4"
  },
  {
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_26",
    "name": "book_blue_thick_hardcover_on_white_shelf_platform_26",
    "category": "book",
    "centroid": {
      "x": 0.582,
      "y": -5.3909,
      "z": 1.438
    },
    "heading": {
      "hx": 0.0017,
      "hy": 1.0
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.674,
        "y": -5.5246
      },
      {
        "x": 0.483,
        "y": -5.5243
      },
      {
        "x": 0.4835,
        "y": -5.258
      },
      {
        "x": 0.6745,
        "y": -5.2583
      }
    ],
    "z_range": [
      1.3984,
      1.4773
    ],
    "size": {
      "x_length": 0.1914,
      "y_length": 0.2666,
      "z_length": 0.0789
    },
    "on_platform": "frl_apartment_wall_cabinet_02_21_4"
  },
  {
    "object_id": "kitchenware_dark_blue_saucepan_on_wooden_shelf_43",
    "name": "kitchenware_dark_blue_saucepan_on_wooden_shelf_43",
    "category": "kitchenware",
    "centroid": {
      "x": 0.507,
      "y": -2.2998,
      "z": 0.2653
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.319,
        "y": -2.3921
      },
      {
        "x": 0.4839,
        "y": -2.1631
      },
      {
        "x": 0.6738,
        "y": -2.2998
      },
      {
        "x": 0.5089,
        "y": -2.5288
      }
    ],
    "z_range": [
      0.2255,
      0.3358
    ],
    "size": {
      "x_length": 0.3548,
      "y_length": 0.3657,
      "z_length": 0.1103
    },
    "on_platform": "frl_apartment_table_02_40_1"
  },
  {
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
    "name": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
    "category": "kitchenware",
    "centroid": {
      "x": 1.0512,
      "y": -2.6802,
      "z": 0.2312
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.8244,
        "y": -2.688
      },
      {
        "x": 0.9867,
        "y": -2.4627
      },
      {
        "x": 1.278,
        "y": -2.6724
      },
      {
        "x": 1.1157,
        "y": -2.8977
      }
    ],
    "z_range": [
      0.2244,
      0.2379
    ],
    "size": {
      "x_length": 0.4535,
      "y_length": 0.435,
      "z_length": 0.0136
    },
    "on_platform": "frl_apartment_table_02_40_2"
  },
  {
    "object_id": "kitchenware_gray_cast_iron_casserole_on_wooden_platform_44",
    "name": "kitchenware_gray_cast_iron_casserole_on_wooden_platform_44",
    "category": "kitchenware",
    "centroid": {
      "x": 1.0548,
      "y": -2.6772,
      "z": 0.304
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.8313,
        "y": -2.6835
      },
      {
        "x": 0.9899,
        "y": -2.4633
      },
      {
        "x": 1.2785,
        "y": -2.6711
      },
      {
        "x": 1.1199,
        "y": -2.8913
      }
    ],
    "z_range": [
      0.2406,
      0.4025
    ],
    "size": {
      "x_length": 0.4472,
      "y_length": 0.428,
      "z_length": 0.1619
    },
    "on_platform": "frl_apartment_table_02_40_2"
  },
  {
    "object_id": "furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
    "name": "furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
    "category": "furniture",
    "centroid": {
      "x": 0.9771,
      "y": -2.6448,
      "z": 0.646
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.6092,
        "y": -2.5693
      },
      {
        "x": 0.7887,
        "y": -2.3198
      },
      {
        "x": 1.3446,
        "y": -2.72
      },
      {
        "x": 1.165,
        "y": -2.9695
      }
    ],
    "z_range": [
      0.5722,
      0.7018
    ],
    "size": {
      "x_length": 0.7354,
      "y_length": 0.6496,
      "z_length": 0.1297
    },
    "on_platform": "frl_apartment_table_02_40_3"
  },
  {
    "object_id": "kitchenware_gray_shallow_bowl_on_wooden_shelf_41",
    "name": "kitchenware_gray_shallow_bowl_on_wooden_shelf_41",
    "category": "kitchenware",
    "centroid": {
      "x": 0.524,
      "y": -2.3004,
      "z": 0.6429
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.2953,
        "y": -2.3377
      },
      {
        "x": 0.4868,
        "y": -2.0717
      },
      {
        "x": 0.7526,
        "y": -2.2631
      },
      {
        "x": 0.5611,
        "y": -2.5291
      }
    ],
    "z_range": [
      0.5727,
      0.7013
    ],
    "size": {
      "x_length": 0.4573,
      "y_length": 0.4573,
      "z_length": 0.1286
    },
    "on_platform": "frl_apartment_table_02_40_3"
  },
  {
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
    "name": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
    "category": "decor",
    "centroid": {
      "x": 0.6498,
      "y": -2.1779,
      "z": 1.1334
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.4832,
        "y": -2.202
      },
      {
        "x": 0.6196,
        "y": -2.0125
      },
      {
        "x": 0.8162,
        "y": -2.1541
      },
      {
        "x": 0.6798,
        "y": -2.3435
      }
    ],
    "z_range": [
      0.9389,
      1.2852
    ],
    "size": {
      "x_length": 0.333,
      "y_length": 0.331,
      "z_length": 0.3463
    },
    "on_platform": "frl_apartment_table_02_40_4"
  },
  {
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82",
    "name": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82",
    "category": "decor",
    "centroid": {
      "x": 1.1438,
      "y": -2.5775,
      "z": 1.1334
    },
    "heading": {
      "hx": 0.8115,
      "hy": -0.5843
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 0.9772,
        "y": -2.6016
      },
      {
        "x": 1.1136,
        "y": -2.4121
      },
      {
        "x": 1.3101,
        "y": -2.5536
      },
      {
        "x": 1.1737,
        "y": -2.7431
      }
    ],
    "z_range": [
      0.9389,
      1.2853
    ],
    "size": {
      "x_length": 0.333,
      "y_length": 0.331,
      "z_length": 0.3464
    },
    "on_platform": "frl_apartment_table_02_40_4"
  },
  {
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_20",
    "name": "electronics_black_wireless_mouse_on_blue_table_platform_20",
    "category": "electronics",
    "centroid": {
      "x": 2.8197,
      "y": -7.6627,
      "z": 0.6221
    },
    "heading": {
      "hx": -1.0,
      "hy": 0.0009
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 2.8431,
        "y": -7.5741
      },
      {
        "x": 2.8429,
        "y": -7.7514
      },
      {
        "x": 2.7963,
        "y": -7.7514
      },
      {
        "x": 2.7964,
        "y": -7.574
      }
    ],
    "z_range": [
      0.6157,
      0.6286
    ],
    "size": {
      "x_length": 0.0468,
      "y_length": 0.1774,
      "z_length": 0.0129
    },
    "on_platform": "frl_apartment_tvstand_89_0"
  },
  {
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_19",
    "name": "electronics_black_wireless_mouse_on_blue_table_platform_19",
    "category": "electronics",
    "centroid": {
      "x": 2.9747,
      "y": -7.6124,
      "z": 0.6221
    },
    "heading": {
      "hx": -1.0,
      "hy": 0.0009
    },
    "quaternion": null,
    "bbox_2d": [
      {
        "x": 3.0198,
        "y": -7.5273
      },
      {
        "x": 3.0197,
        "y": -7.6975
      },
      {
        "x": 2.9297,
        "y": -7.6974
      },
      {
        "x": 2.9298,
        "y": -7.5272
      }
    ],
    "z_range": [
      0.6157,
      0.6285
    ],
    "size": {
      "x_length": 0.0902,
      "y_length": 0.1702,
      "z_length": 0.0129
    },
    "on_platform": "frl_apartment_tvstand_89_0"
  }
]
}

SUPPORT_RELATIONS:
[
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_11",
    "supported_by_platform_id": "frl_apartment_sofa_10_1"
  },
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_12",
    "supported_by_platform_id": "frl_apartment_sofa_10_1"
  },
  {
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_9",
    "supported_by_platform_id": "frl_apartment_sofa_10_2"
  },
  {
    "object_id": "furniture_gray_tissue_box_on_platform_17",
    "supported_by_platform_id": "frl_apartment_stool_02_18_0"
  },
  {
    "object_id": "footwear_red_high_heel_on_shelf_platform_79",
    "supported_by_platform_id": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "footwear_black_sneaker_on_platform_77",
    "supported_by_platform_id": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "footwear_brown_leather_dress_shoe_on_platform_78",
    "supported_by_platform_id": "frl_apartment_rack_01_76_1"
  },
  {
    "object_id": "electronics_black_wall_bracket_on_platform_2",
    "supported_by_platform_id": "kitchen_counter_1_body_0"
  },
  {
    "object_id": "kitchenware_white_cylindrical_spice_jars_on_platform_59",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_66",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_67",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_65",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_64",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_62",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_60",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_61",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_63",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "decor_photo_frame_with_dog_picture_on_platform_70",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_octagonal_plate_on_black_platform_50",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_beige_coffee_cup_on_black_table_platform_51",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_flower_patterned_cream_jug_on_platform_55",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_shallow_bowl_on_table_platform_57",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_small_coffee_mug_on_black_table_platform_52",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_gray_cylindrical_container_on_platform_72",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_blue_thermos_bottle_73",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_round_bowl_on_black_table_platform_54",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_transparent_bowl_on_platform_56",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_wooden_knife_block_on_platform_71",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "furniture_brown_cushion_on_platform_53",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "container_brown_paper_food_box_on_table_platform_0",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_white_cylindrical_cup_on_black_platform_68",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "kitchenware_brown_black_coffee_grinder_on_table_platform_58",
    "supported_by_platform_id": "kitchen_counter_1_body_2"
  },
  {
    "object_id": "decor_brown_wooden_mantle_clock_on_wall_shelf_5",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_01_4_1"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_88",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_87",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_86",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_01_4_4"
  },
  {
    "object_id": "container_brown_wooden_box_on_white_shelf_84",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_01_4_5"
  },
  {
    "object_id": "electronics_white_security_camera_on_table_platform_49",
    "supported_by_platform_id": "frl_apartment_table_01_48_0"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_36",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_37",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_38",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_35",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_0"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_33",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_32",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "book_navy_thick_hardcover_on_platform_34",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_1"
  },
  {
    "object_id": "container_brown_wooden_box_on_white_shelf_85",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_2"
  },
  {
    "object_id": "book_black_thin_softcover_on_table_platform_25",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_39",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "decor_abstract_colorful_painting_on_white_shelf_platform_22",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_24",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_navy_thick_hardcover_on_platform_23",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_29",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_31",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_28",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_30",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_3"
  },
  {
    "object_id": "book_white_thick_hardcover_on_table_platform_27",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_4"
  },
  {
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_26",
    "supported_by_platform_id": "frl_apartment_wall_cabinet_02_21_4"
  },
  {
    "object_id": "kitchenware_dark_blue_saucepan_on_wooden_shelf_43",
    "supported_by_platform_id": "frl_apartment_table_02_40_1"
  },
  {
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
    "supported_by_platform_id": "frl_apartment_table_02_40_2"
  },
  {
    "object_id": "kitchenware_gray_cast_iron_casserole_on_wooden_platform_44",
    "supported_by_platform_id": "frl_apartment_table_02_40_2"
  },
  {
    "object_id": "furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
    "supported_by_platform_id": "frl_apartment_table_02_40_3"
  },
  {
    "object_id": "kitchenware_gray_shallow_bowl_on_wooden_shelf_41",
    "supported_by_platform_id": "frl_apartment_table_02_40_3"
  },
  {
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
    "supported_by_platform_id": "frl_apartment_table_02_40_4"
  },
  {
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82",
    "supported_by_platform_id": "frl_apartment_table_02_40_4"
  },
  {
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_20",
    "supported_by_platform_id": "frl_apartment_tvstand_89_0"
  },
  {
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_19",
    "supported_by_platform_id": "frl_apartment_tvstand_89_0"
  }
]

PLATFORM_IMAGE_MANIFEST:
[
  {
    "image_id": "img_p01",
    "platform_id": "frl_apartment_sofa_10_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform sofa_10_platform_0",
    "path": "images/img_p01_frl_apartment_sofa_10_0.png"
  },
  {
    "image_id": "img_p02",
    "platform_id": "frl_apartment_sofa_10_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform sofa_10_platform_1",
    "path": "images/img_p02_frl_apartment_sofa_10_1.png"
  },
  {
    "image_id": "img_p03",
    "platform_id": "frl_apartment_sofa_10_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform sofa_10_platform_2",
    "path": "images/img_p03_frl_apartment_sofa_10_2.png"
  },
  {
    "image_id": "img_p04",
    "platform_id": "frl_apartment_shoe_04_80_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform shoe_04_80_platform_0",
    "path": "images/img_p04_frl_apartment_shoe_04_80_0.png"
  },
  {
    "image_id": "img_p05",
    "platform_id": "frl_apartment_shoe_04_80_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform shoe_04_80_platform_1",
    "path": "images/img_p05_frl_apartment_shoe_04_80_1.png"
  },
  {
    "image_id": "img_p06",
    "platform_id": "frl_apartment_table_04_13_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_04_13_platform_0",
    "path": "images/img_p06_frl_apartment_table_04_13_0.png"
  },
  {
    "image_id": "img_p07",
    "platform_id": "frl_apartment_chair_04_46_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_04_46_platform_0",
    "path": "images/img_p07_frl_apartment_chair_04_46_0.png"
  },
  {
    "image_id": "img_p08",
    "platform_id": "frl_apartment_chair_05_8_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_05_8_platform_0",
    "path": "images/img_p08_frl_apartment_chair_05_8_0.png"
  },
  {
    "image_id": "img_p09",
    "platform_id": "frl_apartment_chair_05_7_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_05_7_platform_0",
    "path": "images/img_p09_frl_apartment_chair_05_7_0.png"
  },
  {
    "image_id": "img_p10",
    "platform_id": "frl_apartment_stool_02_18_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform stool_02_18_platform_0",
    "path": "images/img_p10_frl_apartment_stool_02_18_0.png"
  },
  {
    "image_id": "img_p11",
    "platform_id": "frl_apartment_stool_02_6_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform stool_02_6_platform_0",
    "path": "images/img_p11_frl_apartment_stool_02_6_0.png"
  },
  {
    "image_id": "img_p12",
    "platform_id": "frl_apartment_rack_01_76_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform rack_01_76_platform_0",
    "path": "images/img_p12_frl_apartment_rack_01_76_0.png"
  },
  {
    "image_id": "img_p13",
    "platform_id": "frl_apartment_rack_01_76_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform rack_01_76_platform_1",
    "path": "images/img_p13_frl_apartment_rack_01_76_1.png"
  },
  {
    "image_id": "img_p14",
    "platform_id": "frl_apartment_rack_01_76_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform rack_01_76_platform_2",
    "path": "images/img_p14_frl_apartment_rack_01_76_2.png"
  },
  {
    "image_id": "img_p15",
    "platform_id": "kitchen_counter_1_body_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform kitchen_counter_1_body_platform_0",
    "path": "images/img_p15_kitchen_counter_1_body_0.png"
  },
  {
    "image_id": "img_p16",
    "platform_id": "kitchen_counter_1_body_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform kitchen_counter_1_body_platform_1",
    "path": "images/img_p16_kitchen_counter_1_body_1.png"
  },
  {
    "image_id": "img_p17",
    "platform_id": "kitchen_counter_1_body_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform kitchen_counter_1_body_platform_2",
    "path": "images/img_p17_kitchen_counter_1_body_2.png"
  },
  {
    "image_id": "img_p18",
    "platform_id": "fridge_0_body_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_1",
    "path": "images/img_p18_fridge_0_body_1.png"
  },
  {
    "image_id": "img_p19",
    "platform_id": "fridge_0_body_3",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_3",
    "path": "images/img_p19_fridge_0_body_3.png"
  },
  {
    "image_id": "img_p20",
    "platform_id": "fridge_0_body_4",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_4",
    "path": "images/img_p20_fridge_0_body_4.png"
  },
  {
    "image_id": "img_p21",
    "platform_id": "fridge_0_body_5",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_5",
    "path": "images/img_p21_fridge_0_body_5.png"
  },
  {
    "image_id": "img_p22",
    "platform_id": "fridge_0_body_6",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_6",
    "path": "images/img_p22_fridge_0_body_6.png"
  },
  {
    "image_id": "img_p23",
    "platform_id": "fridge_0_body_7",
    "view_id": "human_full",
    "view_description": "clean full overview of platform fridge_0_body_platform_7",
    "path": "images/img_p23_fridge_0_body_7.png"
  },
  {
    "image_id": "img_p24",
    "platform_id": "frl_apartment_bin_03_3_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform bin_03_3_platform_0",
    "path": "images/img_p24_frl_apartment_bin_03_3_0.png"
  },
  {
    "image_id": "img_p25",
    "platform_id": "frl_apartment_wall_cabinet_01_4_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_0",
    "path": "images/img_p25_frl_apartment_wall_cabinet_01_4_0.png"
  },
  {
    "image_id": "img_p26",
    "platform_id": "frl_apartment_wall_cabinet_01_4_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_1",
    "path": "images/img_p26_frl_apartment_wall_cabinet_01_4_1.png"
  },
  {
    "image_id": "img_p27",
    "platform_id": "frl_apartment_wall_cabinet_01_4_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_2",
    "path": "images/img_p27_frl_apartment_wall_cabinet_01_4_2.png"
  },
  {
    "image_id": "img_p28",
    "platform_id": "frl_apartment_wall_cabinet_01_4_3",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_3",
    "path": "images/img_p28_frl_apartment_wall_cabinet_01_4_3.png"
  },
  {
    "image_id": "img_p29",
    "platform_id": "frl_apartment_wall_cabinet_01_4_4",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_4",
    "path": "images/img_p29_frl_apartment_wall_cabinet_01_4_4.png"
  },
  {
    "image_id": "img_p30",
    "platform_id": "frl_apartment_wall_cabinet_01_4_5",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_5",
    "path": "images/img_p30_frl_apartment_wall_cabinet_01_4_5.png"
  },
  {
    "image_id": "img_p31",
    "platform_id": "frl_apartment_wall_cabinet_01_4_6",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_01_4_platform_6",
    "path": "images/img_p31_frl_apartment_wall_cabinet_01_4_6.png"
  },
  {
    "image_id": "img_p32",
    "platform_id": "frl_apartment_table_03_14_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_03_14_platform_1",
    "path": "images/img_p32_frl_apartment_table_03_14_1.png"
  },
  {
    "image_id": "img_p33",
    "platform_id": "frl_apartment_table_01_48_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_01_48_platform_0",
    "path": "images/img_p33_frl_apartment_table_01_48_0.png"
  },
  {
    "image_id": "img_p34",
    "platform_id": "frl_apartment_wall_cabinet_02_21_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_0",
    "path": "images/img_p34_frl_apartment_wall_cabinet_02_21_0.png"
  },
  {
    "image_id": "img_p35",
    "platform_id": "frl_apartment_wall_cabinet_02_21_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_1",
    "path": "images/img_p35_frl_apartment_wall_cabinet_02_21_1.png"
  },
  {
    "image_id": "img_p36",
    "platform_id": "frl_apartment_wall_cabinet_02_21_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_2",
    "path": "images/img_p36_frl_apartment_wall_cabinet_02_21_2.png"
  },
  {
    "image_id": "img_p37",
    "platform_id": "frl_apartment_wall_cabinet_02_21_3",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_3",
    "path": "images/img_p37_frl_apartment_wall_cabinet_02_21_3.png"
  },
  {
    "image_id": "img_p38",
    "platform_id": "frl_apartment_wall_cabinet_02_21_4",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_4",
    "path": "images/img_p38_frl_apartment_wall_cabinet_02_21_4.png"
  },
  {
    "image_id": "img_p39",
    "platform_id": "frl_apartment_wall_cabinet_02_21_5",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_5",
    "path": "images/img_p39_frl_apartment_wall_cabinet_02_21_5.png"
  },
  {
    "image_id": "img_p40",
    "platform_id": "frl_apartment_wall_cabinet_02_21_6",
    "view_id": "human_full",
    "view_description": "clean full overview of platform wall_cabinet_02_21_platform_6",
    "path": "images/img_p40_frl_apartment_wall_cabinet_02_21_6.png"
  },
  {
    "image_id": "img_p41",
    "platform_id": "frl_apartment_chair_01_15_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_01_15_platform_1",
    "path": "images/img_p41_frl_apartment_chair_01_15_1.png"
  },
  {
    "image_id": "img_p42",
    "platform_id": "frl_apartment_chair_01_15_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_01_15_platform_2",
    "path": "images/img_p42_frl_apartment_chair_01_15_2.png"
  },
  {
    "image_id": "img_p43",
    "platform_id": "frl_apartment_chair_01_16_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_01_16_platform_1",
    "path": "images/img_p43_frl_apartment_chair_01_16_1.png"
  },
  {
    "image_id": "img_p44",
    "platform_id": "frl_apartment_chair_01_16_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_01_16_platform_2",
    "path": "images/img_p44_frl_apartment_chair_01_16_2.png"
  },
  {
    "image_id": "img_p45",
    "platform_id": "chestOfDrawers_01_3_body_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_0",
    "path": "images/img_p45_chestOfDrawers_01_3_body_0.png"
  },
  {
    "image_id": "img_p46",
    "platform_id": "chestOfDrawers_01_3_body_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_1",
    "path": "images/img_p46_chestOfDrawers_01_3_body_1.png"
  },
  {
    "image_id": "img_p47",
    "platform_id": "chestOfDrawers_01_3_body_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_2",
    "path": "images/img_p47_chestOfDrawers_01_3_body_2.png"
  },
  {
    "image_id": "img_p48",
    "platform_id": "chestOfDrawers_01_3_body_3",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_3",
    "path": "images/img_p48_chestOfDrawers_01_3_body_3.png"
  },
  {
    "image_id": "img_p49",
    "platform_id": "chestOfDrawers_01_3_body_4",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_4",
    "path": "images/img_p49_chestOfDrawers_01_3_body_4.png"
  },
  {
    "image_id": "img_p50",
    "platform_id": "chestOfDrawers_01_3_body_5",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_5",
    "path": "images/img_p50_chestOfDrawers_01_3_body_5.png"
  },
  {
    "image_id": "img_p51",
    "platform_id": "chestOfDrawers_01_3_body_6",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chestOfDrawers_01_3_body_platform_6",
    "path": "images/img_p51_chestOfDrawers_01_3_body_6.png"
  },
  {
    "image_id": "img_p52",
    "platform_id": "frl_apartment_chair_04_47_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform chair_04_47_platform_0",
    "path": "images/img_p52_frl_apartment_chair_04_47_0.png"
  },
  {
    "image_id": "img_p53",
    "platform_id": "frl_apartment_table_02_40_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_02_40_platform_0",
    "path": "images/img_p53_frl_apartment_table_02_40_0.png"
  },
  {
    "image_id": "img_p54",
    "platform_id": "frl_apartment_table_02_40_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_02_40_platform_1",
    "path": "images/img_p54_frl_apartment_table_02_40_1.png"
  },
  {
    "image_id": "img_p55",
    "platform_id": "frl_apartment_table_02_40_2",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_02_40_platform_2",
    "path": "images/img_p55_frl_apartment_table_02_40_2.png"
  },
  {
    "image_id": "img_p56",
    "platform_id": "frl_apartment_table_02_40_3",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_02_40_platform_3",
    "path": "images/img_p56_frl_apartment_table_02_40_3.png"
  },
  {
    "image_id": "img_p57",
    "platform_id": "frl_apartment_table_02_40_4",
    "view_id": "human_full",
    "view_description": "clean full overview of platform table_02_40_platform_4",
    "path": "images/img_p57_frl_apartment_table_02_40_4.png"
  },
  {
    "image_id": "img_p58",
    "platform_id": "frl_apartment_tvstand_89_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform tvstand_89_platform_0",
    "path": "images/img_p58_frl_apartment_tvstand_89_0.png"
  },
  {
    "image_id": "img_p59",
    "platform_id": "cabinet_4_body_0",
    "view_id": "human_full",
    "view_description": "clean full overview of platform cabinet_4_body_platform_0",
    "path": "images/img_p59_cabinet_4_body_0.png"
  },
  {
    "image_id": "img_p60",
    "platform_id": "cabinet_4_body_1",
    "view_id": "human_full",
    "view_description": "clean full overview of platform cabinet_4_body_platform_1",
    "path": "images/img_p60_cabinet_4_body_1.png"
  }
]

OBJECT_IMAGE_MANIFEST:
[
  {
    "image_id": "img_o01",
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_11",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_light_blue_square_cushion_on_sofa_platform_11",
    "path": "images/img_o01_furniture_light_blue_square_cushion_on_sofa_platform_11.png"
  },
  {
    "image_id": "img_o02",
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_12",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_light_blue_square_cushion_on_sofa_platform_12",
    "path": "images/img_o02_furniture_light_blue_square_cushion_on_sofa_platform_12.png"
  },
  {
    "image_id": "img_o03",
    "object_id": "furniture_light_blue_square_cushion_on_sofa_platform_9",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_light_blue_square_cushion_on_sofa_platform_9",
    "path": "images/img_o03_furniture_light_blue_square_cushion_on_sofa_platform_9.png"
  },
  {
    "image_id": "img_o04",
    "object_id": "furniture_gray_tissue_box_on_platform_17",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_gray_tissue_box_on_platform_17",
    "path": "images/img_o04_furniture_gray_tissue_box_on_platform_17.png"
  },
  {
    "image_id": "img_o05",
    "object_id": "footwear_red_high_heel_on_shelf_platform_79",
    "view_id": "human_focus",
    "view_description": "clean close-up of object footwear_red_high_heel_on_shelf_platform_79",
    "path": "images/img_o05_footwear_red_high_heel_on_shelf_platform_79.png"
  },
  {
    "image_id": "img_o06",
    "object_id": "footwear_black_sneaker_on_platform_77",
    "view_id": "human_focus",
    "view_description": "clean close-up of object footwear_black_sneaker_on_platform_77",
    "path": "images/img_o06_footwear_black_sneaker_on_platform_77.png"
  },
  {
    "image_id": "img_o07",
    "object_id": "footwear_brown_leather_dress_shoe_on_platform_78",
    "view_id": "human_focus",
    "view_description": "clean close-up of object footwear_brown_leather_dress_shoe_on_platform_78",
    "path": "images/img_o07_footwear_brown_leather_dress_shoe_on_platform_78.png"
  },
  {
    "image_id": "img_o08",
    "object_id": "electronics_black_wall_bracket_on_platform_2",
    "view_id": "human_focus",
    "view_description": "clean close-up of object electronics_black_wall_bracket_on_platform_2",
    "path": "images/img_o08_electronics_black_wall_bracket_on_platform_2.png"
  },
  {
    "image_id": "img_o09",
    "object_id": "kitchenware_white_cylindrical_spice_jars_on_platform_59",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_cylindrical_spice_jars_on_platform_59",
    "path": "images/img_o09_kitchenware_white_cylindrical_spice_jars_on_platform_59.png"
  },
  {
    "image_id": "img_o10",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_66",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_66",
    "path": "images/img_o10_kitchenware_orange_spice_shaker_on_white_platform_66.png"
  },
  {
    "image_id": "img_o11",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_67",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_67",
    "path": "images/img_o11_kitchenware_orange_spice_shaker_on_white_platform_67.png"
  },
  {
    "image_id": "img_o12",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_65",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_65",
    "path": "images/img_o12_kitchenware_orange_spice_shaker_on_white_platform_65.png"
  },
  {
    "image_id": "img_o13",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_64",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_64",
    "path": "images/img_o13_kitchenware_orange_spice_shaker_on_white_platform_64.png"
  },
  {
    "image_id": "img_o14",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_62",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_62",
    "path": "images/img_o14_kitchenware_orange_spice_shaker_on_white_platform_62.png"
  },
  {
    "image_id": "img_o15",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_60",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_60",
    "path": "images/img_o15_kitchenware_orange_spice_shaker_on_white_platform_60.png"
  },
  {
    "image_id": "img_o16",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_61",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_61",
    "path": "images/img_o16_kitchenware_orange_spice_shaker_on_white_platform_61.png"
  },
  {
    "image_id": "img_o17",
    "object_id": "kitchenware_orange_spice_shaker_on_white_platform_63",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_orange_spice_shaker_on_white_platform_63",
    "path": "images/img_o17_kitchenware_orange_spice_shaker_on_white_platform_63.png"
  },
  {
    "image_id": "img_o18",
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_wooden_cutting_board_on_black_table_platform_69",
    "path": "images/img_o18_kitchenware_brown_wooden_cutting_board_on_black_table_platform_69.png"
  },
  {
    "image_id": "img_o19",
    "object_id": "decor_photo_frame_with_dog_picture_on_platform_70",
    "view_id": "human_focus",
    "view_description": "clean close-up of object decor_photo_frame_with_dog_picture_on_platform_70",
    "path": "images/img_o19_decor_photo_frame_with_dog_picture_on_platform_70.png"
  },
  {
    "image_id": "img_o20",
    "object_id": "kitchenware_white_octagonal_plate_on_black_platform_50",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_octagonal_plate_on_black_platform_50",
    "path": "images/img_o20_kitchenware_white_octagonal_plate_on_black_platform_50.png"
  },
  {
    "image_id": "img_o21",
    "object_id": "kitchenware_beige_coffee_cup_on_black_table_platform_51",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_beige_coffee_cup_on_black_table_platform_51",
    "path": "images/img_o21_kitchenware_beige_coffee_cup_on_black_table_platform_51.png"
  },
  {
    "image_id": "img_o22",
    "object_id": "kitchenware_white_flower_patterned_cream_jug_on_platform_55",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_flower_patterned_cream_jug_on_platform_55",
    "path": "images/img_o22_kitchenware_white_flower_patterned_cream_jug_on_platform_55.png"
  },
  {
    "image_id": "img_o23",
    "object_id": "kitchenware_white_shallow_bowl_on_table_platform_57",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_shallow_bowl_on_table_platform_57",
    "path": "images/img_o23_kitchenware_white_shallow_bowl_on_table_platform_57.png"
  },
  {
    "image_id": "img_o24",
    "object_id": "kitchenware_white_small_coffee_mug_on_black_table_platform_52",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_small_coffee_mug_on_black_table_platform_52",
    "path": "images/img_o24_kitchenware_white_small_coffee_mug_on_black_table_platform_52.png"
  },
  {
    "image_id": "img_o25",
    "object_id": "kitchenware_gray_cylindrical_container_on_platform_72",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_gray_cylindrical_container_on_platform_72",
    "path": "images/img_o25_kitchenware_gray_cylindrical_container_on_platform_72.png"
  },
  {
    "image_id": "img_o26",
    "object_id": "kitchenware_blue_thermos_bottle_73",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_blue_thermos_bottle_73",
    "path": "images/img_o26_kitchenware_blue_thermos_bottle_73.png"
  },
  {
    "image_id": "img_o27",
    "object_id": "kitchenware_white_round_bowl_on_black_table_platform_54",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_round_bowl_on_black_table_platform_54",
    "path": "images/img_o27_kitchenware_white_round_bowl_on_black_table_platform_54.png"
  },
  {
    "image_id": "img_o28",
    "object_id": "kitchenware_transparent_bowl_on_platform_56",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_transparent_bowl_on_platform_56",
    "path": "images/img_o28_kitchenware_transparent_bowl_on_platform_56.png"
  },
  {
    "image_id": "img_o29",
    "object_id": "kitchenware_wooden_knife_block_on_platform_71",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_wooden_knife_block_on_platform_71",
    "path": "images/img_o29_kitchenware_wooden_knife_block_on_platform_71.png"
  },
  {
    "image_id": "img_o30",
    "object_id": "furniture_brown_cushion_on_platform_53",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_brown_cushion_on_platform_53",
    "path": "images/img_o30_furniture_brown_cushion_on_platform_53.png"
  },
  {
    "image_id": "img_o31",
    "object_id": "kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74",
    "path": "images/img_o31_kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74.png"
  },
  {
    "image_id": "img_o32",
    "object_id": "container_brown_paper_food_box_on_table_platform_0",
    "view_id": "human_focus",
    "view_description": "clean close-up of object container_brown_paper_food_box_on_table_platform_0",
    "path": "images/img_o32_container_brown_paper_food_box_on_table_platform_0.png"
  },
  {
    "image_id": "img_o33",
    "object_id": "kitchenware_white_cylindrical_cup_on_black_platform_68",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_white_cylindrical_cup_on_black_platform_68",
    "path": "images/img_o33_kitchenware_white_cylindrical_cup_on_black_platform_68.png"
  },
  {
    "image_id": "img_o34",
    "object_id": "kitchenware_brown_black_coffee_grinder_on_table_platform_58",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_black_coffee_grinder_on_table_platform_58",
    "path": "images/img_o34_kitchenware_brown_black_coffee_grinder_on_table_platform_58.png"
  },
  {
    "image_id": "img_o35",
    "object_id": "decor_brown_wooden_mantle_clock_on_wall_shelf_5",
    "view_id": "human_focus",
    "view_description": "clean close-up of object decor_brown_wooden_mantle_clock_on_wall_shelf_5",
    "path": "images/img_o35_decor_brown_wooden_mantle_clock_on_wall_shelf_5.png"
  },
  {
    "image_id": "img_o36",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_88",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_88",
    "path": "images/img_o36_book_maroon_thin_hardcover_on_white_shelf_platform_88.png"
  },
  {
    "image_id": "img_o37",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_87",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_87",
    "path": "images/img_o37_book_maroon_thin_hardcover_on_white_shelf_platform_87.png"
  },
  {
    "image_id": "img_o38",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_86",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_86",
    "path": "images/img_o38_book_maroon_thin_hardcover_on_white_shelf_platform_86.png"
  },
  {
    "image_id": "img_o39",
    "object_id": "container_brown_wooden_box_on_white_shelf_84",
    "view_id": "human_focus",
    "view_description": "clean close-up of object container_brown_wooden_box_on_white_shelf_84",
    "path": "images/img_o39_container_brown_wooden_box_on_white_shelf_84.png"
  },
  {
    "image_id": "img_o40",
    "object_id": "electronics_white_security_camera_on_table_platform_49",
    "view_id": "human_focus",
    "view_description": "clean close-up of object electronics_white_security_camera_on_table_platform_49",
    "path": "images/img_o40_electronics_white_security_camera_on_table_platform_49.png"
  },
  {
    "image_id": "img_o41",
    "object_id": "book_black_thin_softcover_on_table_platform_36",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_black_thin_softcover_on_table_platform_36",
    "path": "images/img_o41_book_black_thin_softcover_on_table_platform_36.png"
  },
  {
    "image_id": "img_o42",
    "object_id": "book_black_thin_softcover_on_table_platform_37",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_black_thin_softcover_on_table_platform_37",
    "path": "images/img_o42_book_black_thin_softcover_on_table_platform_37.png"
  },
  {
    "image_id": "img_o43",
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_38",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_handle_knife_on_white_table_platform_38",
    "path": "images/img_o43_kitchenware_brown_handle_knife_on_white_table_platform_38.png"
  },
  {
    "image_id": "img_o44",
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_35",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_blue_thick_hardcover_on_white_shelf_platform_35",
    "path": "images/img_o44_book_blue_thick_hardcover_on_white_shelf_platform_35.png"
  },
  {
    "image_id": "img_o45",
    "object_id": "book_white_thick_hardcover_on_table_platform_33",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_white_thick_hardcover_on_table_platform_33",
    "path": "images/img_o45_book_white_thick_hardcover_on_table_platform_33.png"
  },
  {
    "image_id": "img_o46",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_32",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_32",
    "path": "images/img_o46_book_maroon_thin_hardcover_on_white_shelf_platform_32.png"
  },
  {
    "image_id": "img_o47",
    "object_id": "book_navy_thick_hardcover_on_platform_34",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_navy_thick_hardcover_on_platform_34",
    "path": "images/img_o47_book_navy_thick_hardcover_on_platform_34.png"
  },
  {
    "image_id": "img_o48",
    "object_id": "container_brown_wooden_box_on_white_shelf_85",
    "view_id": "human_focus",
    "view_description": "clean close-up of object container_brown_wooden_box_on_white_shelf_85",
    "path": "images/img_o48_container_brown_wooden_box_on_white_shelf_85.png"
  },
  {
    "image_id": "img_o49",
    "object_id": "book_black_thin_softcover_on_table_platform_25",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_black_thin_softcover_on_table_platform_25",
    "path": "images/img_o49_book_black_thin_softcover_on_table_platform_25.png"
  },
  {
    "image_id": "img_o50",
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_39",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_handle_knife_on_white_table_platform_39",
    "path": "images/img_o50_kitchenware_brown_handle_knife_on_white_table_platform_39.png"
  },
  {
    "image_id": "img_o51",
    "object_id": "decor_abstract_colorful_painting_on_white_shelf_platform_22",
    "view_id": "human_focus",
    "view_description": "clean close-up of object decor_abstract_colorful_painting_on_white_shelf_platform_22",
    "path": "images/img_o51_decor_abstract_colorful_painting_on_white_shelf_platform_22.png"
  },
  {
    "image_id": "img_o52",
    "object_id": "kitchenware_brown_handle_knife_on_white_table_platform_24",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_handle_knife_on_white_table_platform_24",
    "path": "images/img_o52_kitchenware_brown_handle_knife_on_white_table_platform_24.png"
  },
  {
    "image_id": "img_o53",
    "object_id": "book_navy_thick_hardcover_on_platform_23",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_navy_thick_hardcover_on_platform_23",
    "path": "images/img_o53_book_navy_thick_hardcover_on_platform_23.png"
  },
  {
    "image_id": "img_o54",
    "object_id": "book_white_thick_hardcover_on_table_platform_29",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_white_thick_hardcover_on_table_platform_29",
    "path": "images/img_o54_book_white_thick_hardcover_on_table_platform_29.png"
  },
  {
    "image_id": "img_o55",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_31",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_31",
    "path": "images/img_o55_book_maroon_thin_hardcover_on_white_shelf_platform_31.png"
  },
  {
    "image_id": "img_o56",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_28",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_28",
    "path": "images/img_o56_book_maroon_thin_hardcover_on_white_shelf_platform_28.png"
  },
  {
    "image_id": "img_o57",
    "object_id": "book_maroon_thin_hardcover_on_white_shelf_platform_30",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_maroon_thin_hardcover_on_white_shelf_platform_30",
    "path": "images/img_o57_book_maroon_thin_hardcover_on_white_shelf_platform_30.png"
  },
  {
    "image_id": "img_o58",
    "object_id": "book_white_thick_hardcover_on_table_platform_27",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_white_thick_hardcover_on_table_platform_27",
    "path": "images/img_o58_book_white_thick_hardcover_on_table_platform_27.png"
  },
  {
    "image_id": "img_o59",
    "object_id": "book_blue_thick_hardcover_on_white_shelf_platform_26",
    "view_id": "human_focus",
    "view_description": "clean close-up of object book_blue_thick_hardcover_on_white_shelf_platform_26",
    "path": "images/img_o59_book_blue_thick_hardcover_on_white_shelf_platform_26.png"
  },
  {
    "image_id": "img_o60",
    "object_id": "kitchenware_dark_blue_saucepan_on_wooden_shelf_43",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_dark_blue_saucepan_on_wooden_shelf_43",
    "path": "images/img_o60_kitchenware_dark_blue_saucepan_on_wooden_shelf_43.png"
  },
  {
    "image_id": "img_o61",
    "object_id": "kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_brown_wooden_cutting_board_on_black_table_platform_45",
    "path": "images/img_o61_kitchenware_brown_wooden_cutting_board_on_black_table_platform_45.png"
  },
  {
    "image_id": "img_o62",
    "object_id": "kitchenware_gray_cast_iron_casserole_on_wooden_platform_44",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_gray_cast_iron_casserole_on_wooden_platform_44",
    "path": "images/img_o62_kitchenware_gray_cast_iron_casserole_on_wooden_platform_44.png"
  },
  {
    "image_id": "img_o63",
    "object_id": "furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
    "view_id": "human_focus",
    "view_description": "clean close-up of object furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42",
    "path": "images/img_o63_furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42.png"
  },
  {
    "image_id": "img_o64",
    "object_id": "kitchenware_gray_shallow_bowl_on_wooden_shelf_41",
    "view_id": "human_focus",
    "view_description": "clean close-up of object kitchenware_gray_shallow_bowl_on_wooden_shelf_41",
    "path": "images/img_o64_kitchenware_gray_shallow_bowl_on_wooden_shelf_41.png"
  },
  {
    "image_id": "img_o65",
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
    "view_id": "human_focus",
    "view_description": "clean close-up of object decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81",
    "path": "images/img_o65_decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81.png"
  },
  {
    "image_id": "img_o66",
    "object_id": "decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82",
    "view_id": "human_focus",
    "view_description": "clean close-up of object decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82",
    "path": "images/img_o66_decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82.png"
  },
  {
    "image_id": "img_o67",
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_20",
    "view_id": "human_focus",
    "view_description": "clean close-up of object electronics_black_wireless_mouse_on_blue_table_platform_20",
    "path": "images/img_o67_electronics_black_wireless_mouse_on_blue_table_platform_20.png"
  },
  {
    "image_id": "img_o68",
    "object_id": "electronics_black_wireless_mouse_on_blue_table_platform_19",
    "view_id": "human_focus",
    "view_description": "clean close-up of object electronics_black_wireless_mouse_on_blue_table_platform_19",
    "path": "images/img_o68_electronics_black_wireless_mouse_on_blue_table_platform_19.png"
  }
]

ATTACHED_IMAGES: ['images/img_o01_furniture_light_blue_square_cushion_on_sofa_platform_11.png', 'images/img_o02_furniture_light_blue_square_cushion_on_sofa_platform_12.png', 'images/img_o03_furniture_light_blue_square_cushion_on_sofa_platform_9.png', 'images/img_o04_furniture_gray_tissue_box_on_platform_17.png', 'images/img_o05_footwear_red_high_heel_on_shelf_platform_79.png', 'images/img_o06_footwear_black_sneaker_on_platform_77.png', 'images/img_o07_footwear_brown_leather_dress_shoe_on_platform_78.png', 'images/img_o08_electronics_black_wall_bracket_on_platform_2.png', 'images/img_o09_kitchenware_white_cylindrical_spice_jars_on_platform_59.png', 'images/img_o10_kitchenware_orange_spice_shaker_on_white_platform_66.png', 'images/img_o11_kitchenware_orange_spice_shaker_on_white_platform_67.png', 'images/img_o12_kitchenware_orange_spice_shaker_on_white_platform_65.png', 'images/img_o13_kitchenware_orange_spice_shaker_on_white_platform_64.png', 'images/img_o14_kitchenware_orange_spice_shaker_on_white_platform_62.png', 'images/img_o15_kitchenware_orange_spice_shaker_on_white_platform_60.png', 'images/img_o16_kitchenware_orange_spice_shaker_on_white_platform_61.png', 'images/img_o17_kitchenware_orange_spice_shaker_on_white_platform_63.png', 'images/img_o18_kitchenware_brown_wooden_cutting_board_on_black_table_platform_69.png', 'images/img_o19_decor_photo_frame_with_dog_picture_on_platform_70.png', 'images/img_o20_kitchenware_white_octagonal_plate_on_black_platform_50.png', 'images/img_o21_kitchenware_beige_coffee_cup_on_black_table_platform_51.png', 'images/img_o22_kitchenware_white_flower_patterned_cream_jug_on_platform_55.png', 'images/img_o23_kitchenware_white_shallow_bowl_on_table_platform_57.png', 'images/img_o24_kitchenware_white_small_coffee_mug_on_black_table_platform_52.png', 'images/img_o25_kitchenware_gray_cylindrical_container_on_platform_72.png', 'images/img_o26_kitchenware_blue_thermos_bottle_73.png', 'images/img_o27_kitchenware_white_round_bowl_on_black_table_platform_54.png', 'images/img_o28_kitchenware_transparent_bowl_on_platform_56.png', 'images/img_o29_kitchenware_wooden_knife_block_on_platform_71.png', 'images/img_o30_furniture_brown_cushion_on_platform_53.png', 'images/img_o31_kitchenware_transparent_square_container_with_black_lid_on_black_table_platform_74.png', 'images/img_o32_container_brown_paper_food_box_on_table_platform_0.png', 'images/img_o33_kitchenware_white_cylindrical_cup_on_black_platform_68.png', 'images/img_o34_kitchenware_brown_black_coffee_grinder_on_table_platform_58.png', 'images/img_o35_decor_brown_wooden_mantle_clock_on_wall_shelf_5.png', 'images/img_o36_book_maroon_thin_hardcover_on_white_shelf_platform_88.png', 'images/img_o37_book_maroon_thin_hardcover_on_white_shelf_platform_87.png', 'images/img_o38_book_maroon_thin_hardcover_on_white_shelf_platform_86.png', 'images/img_o39_container_brown_wooden_box_on_white_shelf_84.png', 'images/img_o40_electronics_white_security_camera_on_table_platform_49.png', 'images/img_o41_book_black_thin_softcover_on_table_platform_36.png', 'images/img_o42_book_black_thin_softcover_on_table_platform_37.png', 'images/img_o43_kitchenware_brown_handle_knife_on_white_table_platform_38.png', 'images/img_o44_book_blue_thick_hardcover_on_white_shelf_platform_35.png', 'images/img_o45_book_white_thick_hardcover_on_table_platform_33.png', 'images/img_o46_book_maroon_thin_hardcover_on_white_shelf_platform_32.png', 'images/img_o47_book_navy_thick_hardcover_on_platform_34.png', 'images/img_o48_container_brown_wooden_box_on_white_shelf_85.png', 'images/img_o49_book_black_thin_softcover_on_table_platform_25.png', 'images/img_o50_kitchenware_brown_handle_knife_on_white_table_platform_39.png', 'images/img_o51_decor_abstract_colorful_painting_on_white_shelf_platform_22.png', 'images/img_o52_kitchenware_brown_handle_knife_on_white_table_platform_24.png', 'images/img_o53_book_navy_thick_hardcover_on_platform_23.png', 'images/img_o54_book_white_thick_hardcover_on_table_platform_29.png', 'images/img_o55_book_maroon_thin_hardcover_on_white_shelf_platform_31.png', 'images/img_o56_book_maroon_thin_hardcover_on_white_shelf_platform_28.png', 'images/img_o57_book_maroon_thin_hardcover_on_white_shelf_platform_30.png', 'images/img_o58_book_white_thick_hardcover_on_table_platform_27.png', 'images/img_o59_book_blue_thick_hardcover_on_white_shelf_platform_26.png', 'images/img_o60_kitchenware_dark_blue_saucepan_on_wooden_shelf_43.png', 'images/img_o61_kitchenware_brown_wooden_cutting_board_on_black_table_platform_45.png', 'images/img_o62_kitchenware_gray_cast_iron_casserole_on_wooden_platform_44.png', 'images/img_o63_furniture_navy_blue_curved_cushion_on_wooden_shelf_platform_42.png', 'images/img_o64_kitchenware_gray_shallow_bowl_on_wooden_shelf_41.png', 'images/img_o65_decor_gray_blue_modern_table_lamp_on_wooden_table_platform_81.png', 'images/img_o66_decor_gray_blue_modern_table_lamp_on_wooden_table_platform_82.png', 'images/img_o67_electronics_black_wireless_mouse_on_blue_table_platform_20.png', 'images/img_o68_electronics_black_wireless_mouse_on_blue_table_platform_19.png', 'images/img_p01_frl_apartment_sofa_10_0.png', 'images/img_p02_frl_apartment_sofa_10_1.png', 'images/img_p03_frl_apartment_sofa_10_2.png', 'images/img_p04_frl_apartment_shoe_04_80_0.png', 'images/img_p05_frl_apartment_shoe_04_80_1.png', 'images/img_p06_frl_apartment_table_04_13_0.png', 'images/img_p07_frl_apartment_chair_04_46_0.png', 'images/img_p08_frl_apartment_chair_05_8_0.png', 'images/img_p09_frl_apartment_chair_05_7_0.png', 'images/img_p10_frl_apartment_stool_02_18_0.png', 'images/img_p11_frl_apartment_stool_02_6_0.png', 'images/img_p12_frl_apartment_rack_01_76_0.png', 'images/img_p13_frl_apartment_rack_01_76_1.png', 'images/img_p14_frl_apartment_rack_01_76_2.png', 'images/img_p15_kitchen_counter_1_body_0.png', 'images/img_p16_kitchen_counter_1_body_1.png', 'images/img_p17_kitchen_counter_1_body_2.png', 'images/img_p18_fridge_0_body_1.png', 'images/img_p19_fridge_0_body_3.png', 'images/img_p20_fridge_0_body_4.png', 'images/img_p21_fridge_0_body_5.png', 'images/img_p22_fridge_0_body_6.png', 'images/img_p23_fridge_0_body_7.png', 'images/img_p24_frl_apartment_bin_03_3_0.png', 'images/img_p25_frl_apartment_wall_cabinet_01_4_0.png', 'images/img_p26_frl_apartment_wall_cabinet_01_4_1.png', 'images/img_p27_frl_apartment_wall_cabinet_01_4_2.png', 'images/img_p28_frl_apartment_wall_cabinet_01_4_3.png', 'images/img_p29_frl_apartment_wall_cabinet_01_4_4.png', 'images/img_p30_frl_apartment_wall_cabinet_01_4_5.png', 'images/img_p31_frl_apartment_wall_cabinet_01_4_6.png', 'images/img_p32_frl_apartment_table_03_14_1.png', 'images/img_p33_frl_apartment_table_01_48_0.png', 'images/img_p34_frl_apartment_wall_cabinet_02_21_0.png', 'images/img_p35_frl_apartment_wall_cabinet_02_21_1.png', 'images/img_p36_frl_apartment_wall_cabinet_02_21_2.png', 'images/img_p37_frl_apartment_wall_cabinet_02_21_3.png', 'images/img_p38_frl_apartment_wall_cabinet_02_21_4.png', 'images/img_p39_frl_apartment_wall_cabinet_02_21_5.png', 'images/img_p40_frl_apartment_wall_cabinet_02_21_6.png', 'images/img_p41_frl_apartment_chair_01_15_1.png', 'images/img_p42_frl_apartment_chair_01_15_2.png', 'images/img_p43_frl_apartment_chair_01_16_1.png', 'images/img_p44_frl_apartment_chair_01_16_2.png', 'images/img_p45_chestOfDrawers_01_3_body_0.png', 'images/img_p46_chestOfDrawers_01_3_body_1.png', 'images/img_p47_chestOfDrawers_01_3_body_2.png', 'images/img_p48_chestOfDrawers_01_3_body_3.png', 'images/img_p49_chestOfDrawers_01_3_body_4.png', 'images/img_p50_chestOfDrawers_01_3_body_5.png', 'images/img_p51_chestOfDrawers_01_3_body_6.png', 'images/img_p52_frl_apartment_chair_04_47_0.png', 'images/img_p53_frl_apartment_table_02_40_0.png', 'images/img_p54_frl_apartment_table_02_40_1.png', 'images/img_p55_frl_apartment_table_02_40_2.png', 'images/img_p56_frl_apartment_table_02_40_3.png', 'images/img_p57_frl_apartment_table_02_40_4.png', 'images/img_p58_frl_apartment_tvstand_89_0.png', 'images/img_p59_cabinet_4_body_0.png', 'images/img_p60_cabinet_4_body_1.png']
