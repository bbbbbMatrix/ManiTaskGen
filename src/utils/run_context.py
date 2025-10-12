import os
import glog
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager


class RunContext:
    """Manage runtime context and path resolution"""

    _current_run_dir: Optional[str] = None
    _is_pipeline_mode: bool = False

    @classmethod
    def create_run_dir(cls, custom_name: str = None) -> str:
        """Create a new run directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_name:
            run_name = f"run_{timestamp}_{custom_name}"
        else:
            run_name = f"run_{timestamp}"

        run_dir = f"runs/{run_name}"

        # Create directory structure
        subdirs = [
            "cache",
            "images/image4rename",
            "images/image4voting",
            "images/image4interaction",
            "results",
            "configs",
            "logs",
        ]

        for subdir in subdirs:
            os.makedirs(f"{run_dir}/{subdir}", exist_ok=True)

        # Update the latest symlink
        latest_link = "data/runs/latest"
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(run_name, latest_link)

        glog.info(f"Created run directory: {run_dir}")
        return run_dir

    @classmethod
    def get_or_create_run_dir(
        cls, custom_name: str = None, reuse_existing: bool = False
    ) -> str:
        """Get or create a run directory"""
        if cls._current_run_dir and reuse_existing:
            return cls._current_run_dir

        cls._current_run_dir = cls.create_run_dir(custom_name)
        return cls._current_run_dir

    @classmethod
    @contextmanager
    def pipeline_context(cls, pipeline_name: str = None):
        """Pipeline runtime context, all steps share the same run directory"""
        old_run_dir = cls._current_run_dir
        old_pipeline_mode = cls._is_pipeline_mode

        try:
            cls._is_pipeline_mode = True
            cls._current_run_dir = cls.create_run_dir(pipeline_name or "pipeline")
            glog.info(f"Pipeline context started: {cls._current_run_dir}")
            yield cls._current_run_dir
        finally:
            cls._current_run_dir = old_run_dir
            cls._is_pipeline_mode = old_pipeline_mode

    @classmethod
    def is_pipeline_mode(cls) -> bool:
        return cls._is_pipeline_mode

    @classmethod
    def resolve_path(cls, relative_path: str, run_dir: str = None) -> str:
        """Resolve relative paths to absolute paths, supporting path template substitution"""
        if run_dir is None:
            run_dir = cls._current_run_dir or cls.get_or_create_run_dir()

        # Path template substitution
        path_mappings = {
            "${run_dir}": run_dir,
            "${datasets_root}": "data/datasets",
            "${templates_root}": "data/templates",
            "${cache_root}": "data/cache",
        }

        resolved_path = relative_path
        for template, actual_path in path_mappings.items():
            resolved_path = resolved_path.replace(template, actual_path)

        return resolved_path


class PathResolver:
    """Path resolver to handle different types of paths"""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def resolve(self, path_config: str) -> str:
        """Resolve paths in the configuration"""
        return RunContext.resolve_path(path_config, self.run_dir)

    def get_input_path(self, filename: str) -> str:
        """Get input file path (usually in datasets or templates)"""
        if filename.startswith("data/datasets/") or filename.startswith(
            "data/templates/"
        ):
            return filename
        else:
            # Default to searching in datasets
            return f"data/datasets/{filename}"

    def get_cache_path(self, filename: str) -> str:
        """Get cache file path (in the current run directory)"""
        return f"{self.run_dir}/cache/{filename}"

    def get_image_path(self, subdir: str) -> str:
        """Get image directory path"""
        return f"{self.run_dir}/images/{subdir}"

    def get_result_path(self, filename: str) -> str:
        """Get result file path"""
        return f"{self.run_dir}/results/{filename}"
