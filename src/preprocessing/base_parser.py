"""
Base classes and factory for raw scene parsers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRawSceneParser(ABC):
    """Base class for raw scene parsers"""

    @abstractmethod
    def parse_scene(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Parse raw scene data into standardized format

        Args:
            input_path: Path to input scene file/directory
            output_path: Path to output parsed scene file

        Returns:
            Parsed scene data dictionary
        """
        pass


class RawSceneParserFactory:
    """Factory for creating appropriate scene parsers"""

    _parsers = {}

    @classmethod
    def register_parser(cls, scene_type: str, parser_class):
        """Register a parser class for a specific scene type"""
        cls._parsers[scene_type] = parser_class

    @classmethod
    def create_parser(cls, scene_type: str, config=None):
        """Create appropriate parser based on scene type"""
        if scene_type not in cls._parsers:
            raise ValueError(
                f"Unsupported scene type: {scene_type}. Supported types: {list(cls._parsers.keys())}"
            )
        return cls._parsers[scene_type](config)

    @classmethod
    def get_supported_types(cls):
        """Get list of supported scene types"""
        return list(cls._parsers.keys())
