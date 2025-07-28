"""
utils/string_convertor.py
This module provides functionality to convert and manipulate strings, particularly for extracting names and IDs.

This is mainly used for removing the prefixes(e.g. frl_apartment_) and suffixes from names, handling IDs, and renaming strings based on a mapping.

"""

import numpy as np


class StringConvertor:

    def __init__(self):
        pass

    @staticmethod
    def get_noslash_name_wo_id(name: str) -> str:
        """
        This function takes a string name and removes the ID part from it.
        The ID part is assumed to be the last part of the string, separated by an underscore.
        It also removes any diagonal elements from the name.
        """
        if "/" in name:
            name = name[name.rfind("/") + 1 :]

        return StringConvertor.get_name_wo_id(name)

    @staticmethod
    def get_category(name: str) -> str:
        if "_" in name:
            return name.split("_")[0]
        else:
            return name

    @staticmethod
    def get_name_wo_id(name: str) -> str:
        """
        This function takes a string name and removes the ID part from it.
        The ID part is assumed to be the last part of the string, separated by an underscore.
        """
        if "_" in name:
            return name[: name.rfind("_")]
        else:
            return name

    @staticmethod
    def get_id(name: str) -> str:
        """
        This function takes a string name and extracts the ID part from it.
        The ID part is assumed to be the last part of the string, separated by an underscore.
        """
        if "_" in name:
            return name[name.rfind("_") + 1 :]
        else:
            return None

    @staticmethod
    def rename_with_map(name: str, name_map: dict) -> str:
        """
        This function takes a string name and a dictionary name_map.
        It renames the string name according to the mapping provided in name_map.
        If the name is not found in the map, it returns the original name.
        """
        name_prefix = StringConvertor.get_name_wo_id(name)
        name_suffix = StringConvertor.get_id(name)
        if name_prefix in name_map:
            name_prefix = name_map[name_prefix]

        return f"{name_prefix}_{name_suffix}" if name_suffix else name_prefix
