"""
Configuration Management for HS-Pose Vehicle Estimation
======================================================

This module provides comprehensive configuration management for the project,
including loading from YAML files, command-line argument integration, and
dynamic configuration updates.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch

class Config:
    """
    Configuration class for HS-Pose vehicle pose estimation
    
    This class centralizes all hyperparameters and settings for the project.
    It supports loading from YAML files and command-line argument overrides.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with default values
        
        Args:
            config_dict (dict, optional): Configuration dictionary to load
        """
        # Project metadata
        self.PROJECT_NAME = "HS-Pose Vehicle Estimation"
        self.VERSION = "1.0.0"
        
        # Dataset Configuration
        self.DATASET_ROOT = "./data/kitti"
        self.DATASET_TYPE = "kitti"
        self.MAX_SAMPLES = 100
        self.TRAIN_SPLIT = 0.8
        self.VAL_SPLIT = 0.1
        self.TEST_SPLIT = 0.1
        
        # Model Configuration
        self.INPUT_CHANNELS = 4  # RGB + Depth
        self.NUM_CATEGORIES = 3  # Car, Motorcycle, Bicycle
        self.IMAGE_SIZE = (256, 256)
        self.USE_ORIGINAL_HSPOSE = False
        
        # Training Configuration
        self.BATCH_SIZE = 4
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 50
        self.WEIGHT_DECAY = 1e-4
        self.NUM_WORKERS = 4
        
        # Loss Weights
        self.POSE_LOSS_WEIGHT = 1.0
        self.CATEGORY_LOSS_WEIGHT = 0.5
        self.CONFIDENCE_LOSS_WEIGHT = 0.3
        self.DIMENSION_LOSS_WEIGHT = 0.2
        
        # Evaluation Configuration
        self.TEST_SAMPLES = 50
        self.EVAL_BATCH_SIZE = 1
        
        # Hardware Configuration
        self.DEVICE = "auto"  # auto, cpu, cuda
        self.GPU_IDS = [0]
        self.MIXED_PRECISION = False
        
        # Output Configuration
        self.RESULTS_SAVE_PATH = "./results"
        self.MODEL_SAVE_PATH = "./results/trained_models"
        self.PLOTS_SAVE_PATH = "./results/plots"
        self.LOG_DIR = "./results/logs"
        
        # Logging Configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_INTERVAL = 10
        self.SAVE_CHECKPOINTS = True
        self.CHECKPOINT_INTERVAL = 5
        
        # Comparison Configuration
        self.COMPARE_ARCHITECTURES = False
        self.BASELINE_METHODS = ["rtm3d", "vehipose", "original_hspose"]
        
        # Data Augmentation
        self.USE_AUGMENTATION = True
        self.ROTATION_RANGE = 15  # degrees
        self.BRIGHTNESS_RANGE = 0.2
        
        # Reproducibility
        self.SEED = 42
        self.DETERMINISTIC = True
        
        # Load from dictionary if provided
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary
        
        Args:
            config_dict (dict): Configuration dictionary
        """
        def update_nested(obj, d):
            for key, value in d.items():
                if isinstance(value, dict) and hasattr(obj, key.upper()):
                    # Handle nested configurations
                    nested_obj = getattr(obj, key.upper())
                    if isinstance(nested_obj, dict):
                        nested_obj.update(value)
                    else:
                        update_nested(obj, value)
                else:
                    # Update direct attributes
                    attr_name = key.upper()
                    if hasattr(obj, attr_name):
                        setattr(obj, attr_name, value)
        
        update_nested(self, config_dict)
    
    def update_from_args(self, args: argparse.Namespace):
        """
        Update configuration from command-line arguments
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
        """
        # Map argument names to config attributes
        arg_mapping = {
            'dataset': 'DATASET_ROOT',
            'epochs': 'NUM_EPOCHS',
            'batch_size': 'BATCH_SIZE',
            'lr': 'LEARNING_RATE',
            'model_path': 'MODEL_PATH',
            'use_original_hspose': 'USE_ORIGINAL_HSPOSE',
            'compare_architectures': 'COMPARE_ARCHITECTURES',
            '