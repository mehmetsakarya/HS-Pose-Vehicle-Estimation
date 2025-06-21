"""
Models Package for HS-Pose Vehicle Estimation
============================================

This package contains all model implementations for the HS-Pose vehicle pose estimation project.

Available Models:
- VehiclePoseEstimator: Vehicle-adapted HS-Pose model
- OriginalHSPose: Original HS-Pose implementation for comparison
- VehicleCategories: Vehicle category management utilities

Usage:
    from models import VehiclePoseEstimator, OriginalHSPose, VehicleCategories
    from models.hspose_vehicle import VehiclePoseLoss
    from models.hspose_original import OriginalHSPoseLoss
"""

# Import main model classes
from .hspose_vehicle import (
    VehiclePoseEstimator,
    VehicleCategories,
    VehiclePoseLoss,
    VehicleHybridScopeFeatureExtractor,
    load_vehicle_pose_model,
    create_vehicle_pose_optimizer
)

from .hspose_original import (
    OriginalHSPose,
    OriginalHSPoseLoss,
    OriginalHybridScopeFeatureExtractor,
    load_original_hspose_model,
    convert_vehicle_to_original_format,
    compare_model_architectures
)

# Version information
__version__ = "1.0.0"
__author__ = "HS-Pose Vehicle Estimation Team"

# Model registry for easy access
MODEL_REGISTRY = {
    'vehicle_hspose': VehiclePoseEstimator,
    'original_hspose': OriginalHSPose,
    'hspose_vehicle': VehiclePoseEstimator,  # Alias
    'hspose_original': OriginalHSPose        # Alias
}

# Loss function registry
LOSS_REGISTRY = {
    'vehicle_hspose': VehiclePoseLoss,
    'original_hspose': OriginalHSPoseLoss,
    'hspose_vehicle': VehiclePoseLoss,
    'hspose_original': OriginalHSPoseLoss
}

def create_model(model_name: str, input_channels: int = 4, num_categories: int = 3, **kwargs):
    """
    Factory function to create models by name
    
    Args:
        model_name (str): Name of the model to create
        input_channels (int): Number of input channels
        num_categories (int): Number of categories
        **kwargs: Additional model arguments
    
    Returns:
        nn.Module: Created model instance
    
    Example:
        >>> model = create_model('vehicle_hspose', input_channels=4, num_categories=3)
        >>> original_model = create_model('original_hspose', input_channels=3, num_categories=6)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Adjust parameters based on model type
    if 'original' in model_name:
        # Original HS-Pose expects 3 channels and 6 categories by default
        input_channels = kwargs.get('input_channels', 3)
        num_categories = kwargs.get('num_categories', 6)
    
    return model_class(input_channels=input_channels, num_categories=num_categories, **kwargs)

def create_loss_function(model_name: str, **kwargs):
    """
    Factory function to create loss functions by model name
    
    Args:
        model_name (str): Name of the model
        **kwargs: Loss function arguments
    
    Returns:
        nn.Module: Loss function instance
    
    Example:
        >>> loss_fn = create_loss_function('vehicle_hspose', pose_weight=1.0, category_weight=0.5)
    """
    if model_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown model for loss: {model_name}. Available: {list(LOSS_REGISTRY.keys())}")
    
    loss_class = LOSS_REGISTRY[model_name]
    return loss_class(**kwargs)

def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific model
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Model information
    """
    model_info = {
        'vehicle_hspose': {
            'description': 'Vehicle-adapted HS-Pose for autonomous driving',
            'input_channels': 4,  # RGB + Depth
            'output_categories': 3,  # Car, Motorcycle, Bicycle
            'features': [
                'Multi-modal RGB+LIDAR processing',
                'Vehicle-specific feature extraction',
                'Real-time optimized architecture',
                'Confidence estimation',
                'Dimension prediction'
            ],
            'use_cases': [
                'Autonomous driving',
                'Vehicle tracking',
                'Navigation systems',
                'KITTI dataset evaluation'
            ]
        },
        'original_hspose': {
            'description': 'Original HS-Pose implementation from CVPR 2023',
            'input_channels': 3,  # RGB only
            'output_categories': 6,  # NOCS object categories
            'features': [
                'Hybrid scope feature extraction',
                'NOCS coordinate prediction',
                'Category-level pose estimation',
                'Shape and size estimation'
            ],
            'use_cases': [
                'General object pose estimation',
                'NOCS dataset evaluation',
                'Baseline comparison',
                'Research reference'
            ]
        }
    }
    
    return model_info.get(model_name, {})

def list_available_models() -> list:
    """
    List all available models
    
    Returns:
        list: List of available model names
    """
    return list(MODEL_REGISTRY.keys())

def compare_models(model1_name: str, model2_name: str) -> dict:
    """
    Compare two models
    
    Args:
        model1_name (str): First model name
        model2_name (str): Second model name
    
    Returns:
        dict: Comparison results
    """
    info1 = get_model_info(model1_name)
    info2 = get_model_info(model2_name)
    
    comparison = {
        'model1': {
            'name': model1_name,
            'info': info1
        },
        'model2': {
            'name': model2_name,
            'info': info2
        },
        'differences': {
            'input_channels': info1.get('input_channels', 0) - info2.get('input_channels', 0),
            'output_categories': info1.get('output_categories', 0) - info2.get('output_categories', 0),
        }
    }
    
    return comparison

# Model configuration presets
MODEL_CONFIGS = {
    'vehicle_hspose_default': {
        'input_channels': 4,
        'num_categories': 3,
        'model_name': 'vehicle_hspose'
    },
    'vehicle_hspose_rgb_only': {
        'input_channels': 3,
        'num_categories': 3,
        'model_name': 'vehicle_hspose'
    },
    'original_hspose_default': {
        'input_channels': 3,
        'num_categories': 6,
        'model_name': 'original_hspose'
    },
    'comparison_setup': {
        'vehicle_model': {
            'input_channels': 4,
            'num_categories': 3,
            'model_name': 'vehicle_hspose'
        },
        'original_model': {
            'input_channels': 3,
            'num_categories': 6,
            'model_name': 'original_hspose'
        }
    }
}

def create_model_from_config(config_name: str):
    """
    Create model from predefined configuration
    
    Args:
        config_name (str): Configuration name
    
    Returns:
        nn.Module or dict: Model instance or dict of models
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[config_name]
    
    if config_name == 'comparison_setup':
        # Return both models for comparison
        return {
            'vehicle_model': create_model(**config['vehicle_model']),
            'original_model': create_model(**config['original_model'])
        }
    else:
        # Return single model
        return create_model(**config)

# Export main symbols
__all__ = [
    # Main model classes
    'VehiclePoseEstimator',
    'OriginalHSPose',
    'VehicleCategories',
    
    # Loss functions
    'VehiclePoseLoss',
    'OriginalHSPoseLoss',
    
    # Feature extractors
    'VehicleHybridScopeFeatureExtractor',
    'OriginalHybridScopeFeatureExtractor',
    
    # Utility functions
    'create_model',
    'create_loss_function',
    'get_model_info',
    'list_available_models',
    'compare_models',
    'create_model_from_config',
    
    # Loading functions
    'load_vehicle_pose_model',
    'load_original_hspose_model',
    'create_vehicle_pose_optimizer',
    
    # Comparison utilities
    'convert_vehicle_to_original_format',
    'compare_model_architectures',
    
    # Registries
    'MODEL_REGISTRY',
    'LOSS_REGISTRY',
    'MODEL_CONFIGS'
]

# Package metadata
__package_info__ = {
    'name': 'HS-Pose Vehicle Estimation Models',
    'version': __version__,
    'author': __author__,
    'description': 'Model implementations for vehicle pose estimation using HS-Pose architecture',
    'models': list(MODEL_REGISTRY.keys()),
    'features': [
        'Vehicle-adapted HS-Pose architecture',
        'Original HS-Pose implementation',
        'Multi-modal input processing',
        'Category-level pose estimation',
        'Comprehensive loss functions',
        'Model comparison utilities'
    ]
}

def print_package_info():
    """Print package information"""
    print(" HS-Pose Vehicle Estimation Models Package")
    print("=" * 50)
    print(f"Version: {__package_info__['version']}")
    print(f"Author: {__package_info__['author']}")
    print(f"Description: {__package_info__['description']}")
    
    print(f"\n Available Models:")
    for model_name in __package_info__['models']:
        info = get_model_info(model_name)
        print(f"  • {model_name}: {info.get('description', 'No description')}")
    
    print(f"\n Package Features:")
    for feature in __package_info__['features']:
        print(f"  • {feature}")
    
    print("=" * 50)

# Print info when module is imported
if __name__ == "__main__":
    print_package_info()
    
    # Example usage
    print("\n Example Usage:")
    print("from models import VehiclePoseEstimator, OriginalHSPose")
    print("from models import create_model, get_model_info")
    print("")
    print("# Create vehicle model")
    print("model = create_model('vehicle_hspose')")
    print("")
    print("# Create original model for comparison")
    print("original = create_model('original_hspose')")
    print("")
    print("# Get model information")
    print("info = get_model_info('vehicle_hspose')")
    print("print(info['description'])")

print(" Models package loaded successfully!")
print(f" Available models: {len(MODEL_REGISTRY)}")
print(f"🔧 Available configurations: {len(MODEL_CONFIGS)}")