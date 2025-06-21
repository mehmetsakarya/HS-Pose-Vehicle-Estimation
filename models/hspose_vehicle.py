"""
Vehicle-Specific HS-Pose Adaptation
===================================

This module contains our vehicle-specific adaptation of the HS-Pose architecture
for autonomous driving applications using the KITTI dataset.

Key Adaptations from Original HS-Pose:
1. Multi-modal input processing (RGB + LIDAR depth)
2. Vehicle-specific category definitions (car, motorcycle, bicycle)
3. 6D pose estimation optimized for vehicle navigation
4. KITTI dataset integration with proper calibration handling
5. Real-time performance optimizations

Author: [Your Name]
Course: Computer Vision
Institution: [Your Institution]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .hspose_original import OriginalHybridScopeFeatureExtractor

class VehicleCategories:
    """
    Vehicle category definitions and utility functions
    
    This class defines the vehicle categories for pose estimation and provides
    utility functions for category management and keypoint definitions.
    """
    
    # Vehicle category mapping
    CATEGORIES = {
        'car': 0,
        'motorcycle': 1,
        'bicycle': 2
    }
    
    # Reverse mapping for category names
    CATEGORY_NAMES = {v: k for k, v in CATEGORIES.items()}
    
    # Vehicle-specific keypoints for pose estimation
    VEHICLE_KEYPOINTS = {
        'car': [
            'front_left_corner', 'front_right_corner',
            'rear_left_corner', 'rear_right_corner',
            'front_center', 'rear_center',
            'left_center', 'right_center',
            'roof_center'
        ],
        'motorcycle': [
            'front_wheel_center', 'rear_wheel_center',
            'handlebar_center', 'seat_center',
            'front_fork', 'rear_suspension'
        ],
        'bicycle': [
            'front_wheel_center', 'rear_wheel_center',
            'handlebar_center', 'seat_center',
            'pedal_center', 'frame_center'
        ]
    }
    
    # Typical vehicle dimensions (height, width, length in meters)
    VEHICLE_DIMENSIONS = {
        'car': [1.5, 1.8, 4.2],
        'motorcycle': [1.1, 0.7, 2.1],
        'bicycle': [1.0, 0.6, 1.8]
    }
    
    @classmethod
    def get_category_id(cls, category_name):
        """Get category ID from category name"""
        return cls.CATEGORIES.get(category_name.lower(), -1)
    
    @classmethod
    def get_category_name(cls, category_id):
        """Get category name from category ID"""
        return cls.CATEGORY_NAMES.get(category_id, 'unknown')
    
    @classmethod
    def get_keypoints(cls, category_name):
        """Get keypoints for a specific vehicle category"""
        return cls.VEHICLE_KEYPOINTS.get(category_name.lower(), [])
    
    @classmethod
    def get_dimensions(cls, category_name):
        """Get typical dimensions for a vehicle category"""
        return cls.VEHICLE_DIMENSIONS.get(category_name.lower(), [1.0, 1.0, 1.0])

class VehicleHybridScopeFeatureExtractor(nn.Module):
    """
    Vehicle-adapted Hybrid Scope Feature Extractor
    
    This module extends the original HS-Pose architecture with vehicle-specific
    modifications for autonomous driving applications.
    
    Key Adaptations:
    - 4-channel input processing (RGB + LIDAR depth)
    - Vehicle-optimized feature extraction pathways
    - Automotive-specific prediction heads
    - Enhanced geometric understanding for navigation
    """
    
    def __init__(self, input_channels=4, num_categories=3):
        """
        Initialize the vehicle-adapted HS-Pose feature extractor
        
        Args:
            input_channels (int): Number of input channels (4 for RGB + Depth)
            num_categories (int): Number of vehicle categories (3)
        """
        super(VehicleHybridScopeFeatureExtractor, self).__init__()
        
        self.num_categories = num_categories
        
        # Multi-modal input processing
        # Separate pathways for RGB and depth processing
        self.rgb_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.depth_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Fused input processing
        self.input_fusion = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),  # 32 + 16 = 48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Local scope pathway - adapted for vehicle features
        # Focuses on vehicle-specific details like wheels, edges, lights
        self.local_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.local_bn1 = nn.BatchNorm2d(128)
        self.local_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.local_bn2 = nn.BatchNorm2d(256)
        self.local_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.local_bn3 = nn.BatchNorm2d(512)
        
        # Global scope pathway - adapted for vehicle context
        # Captures vehicle surroundings and scene context
        self.global_conv1 = nn.Conv2d(64, 64, kernel_size=7, padding=3, bias=False)
        self.global_bn1 = nn.BatchNorm2d(64)
        self.global_conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False)
        self.global_bn2 = nn.BatchNorm2d(128)
        self.global_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.global_bn3 = nn.BatchNorm2d(256)
        
        # Vehicle-specific attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512 + 256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature fusion with attention
        self.fusion_conv = nn.Conv2d(512 + 256, 768, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(768)
        
        # Vehicle pose regression head
        # Optimized for 6D vehicle pose estimation
        self.pose_regression = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)  # 6D pose output [tx, ty, tz, rx, ry, rz]
        )
        
        # Vehicle category classification head
        # Specialized for automotive categories
        self.category_classifier = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_categories)
        )
        
        # Confidence estimation head
        # Provides uncertainty quantification for navigation safety
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Vehicle dimension estimation head
        # Estimates 3D bounding box dimensions
        self.dimension_estimator = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # Height, Width, Length
            nn.ReLU(inplace=True)  # Ensure positive dimensions
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the vehicle-adapted HS-Pose network
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, H, W)
                             where channels are [R, G, B, Depth]
        
        Returns:
            dict: Dictionary containing all prediction outputs
        """
        # Split RGB and depth channels
        rgb = x[:, :3, :, :]      # RGB channels
        depth = x[:, 3:4, :, :]   # Depth channel
        
        # Process RGB and depth separately
        rgb_features = self.rgb_processor(rgb)
        depth_features = self.depth_processor(depth)
        
        # Fuse multi-modal features
        fused_input = torch.cat([rgb_features, depth_features], dim=1)
        fused_input = self.input_fusion(fused_input)
        
        # Local scope feature extraction
        local_f1 = torch.relu(self.local_bn1(self.local_conv1(fused_input)))
        local_f2 = torch.relu(self.local_bn2(self.local_conv2(local_f1)))
        local_f3 = torch.relu(self.local_bn3(self.local_conv3(local_f2)))
        
        # Global scope feature extraction
        global_f1 = torch.relu(self.global_bn1(self.global_conv1(fused_input)))
        global_f2 = torch.relu(self.global_bn2(self.global_conv2(global_f1)))
        global_f3 = torch.relu(self.global_bn3(self.global_conv3(global_f2)))
        
        # Feature fusion with attention mechanism
        combined_features = torch.cat([local_f3, global_f3], dim=1)
        attention_weights = self.attention(combined_features)
        attended_features = combined_features * attention_weights
        
        final_features = torch.relu(self.fusion_bn(self.fusion_conv(attended_features)))
        
        # Multi-task predictions
        pose = self.pose_regression(final_features)           # 6D pose
        category = self.category_classifier(final_features)   # Vehicle category
        confidence = self.confidence_estimator(final_features) # Confidence score
        dimensions = self.dimension_estimator(final_features)  # 3D dimensions
        
        return {
            'pose': pose,
            'category': category,
            'confidence': confidence,
            'dimensions': dimensions,
            'features': final_features,
            'attention': attention_weights
        }

class VehiclePoseEstimator(nn.Module):
    """
    Complete vehicle pose estimation model
    
    This is the main model class that wraps the vehicle-adapted HS-Pose feature extractor
    and provides high-level interfaces for training and inference.
    """
    
    def __init__(self, input_channels=4, num_categories=3):
        """
        Initialize the vehicle pose estimation model
        
        Args:
            input_channels (int): Number of input channels (4 for RGB+Depth)
            num_categories (int): Number of vehicle categories (3)
        """
        super(VehiclePoseEstimator, self).__init__()
        
        self.feature_extractor = VehicleHybridScopeFeatureExtractor(input_channels, num_categories)
        self.vehicle_categories = VehicleCategories()
        
        # Pose normalization parameters for stability
        self.pose_mean = torch.tensor([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])  # Typical vehicle distance ~20m
        self.pose_std = torch.tensor([10.0, 5.0, 30.0, 3.14, 1.57, 3.14])  # Reasonable pose ranges
        
    def forward(self, x):
        """Forward pass through the model"""
        return self.feature_extractor(x)
    
    def predict_pose(self, image, category_hint=None):
        """
        Predict 6D pose for a vehicle in the input image
        
        Args:
            image (torch.Tensor or np.ndarray): Input image (RGB+Depth)
            category_hint (str, optional): Hint about vehicle category
        
        Returns:
            dict: Prediction results including pose, category, and confidence
        """
        self.eval()
        with torch.no_grad():
            # Preprocess input
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                elif len(image.shape) == 4 and image.shape[0] > 1:
                    image = image[:1]
            
            # Ensure image is on correct device
            device = next(self.parameters()).device
            image = image.to(device)
            
            # Forward pass
            outputs = self(image)
            
            # Extract pose (6D: translation + rotation)
            pose = outputs['pose']
            if len(pose.shape) > 1:
                pose = pose.squeeze()
            
            # Ensure pose has exactly 6 dimensions
            if len(pose.shape) == 0:
                pose = pose.unsqueeze(0)
            if pose.shape[-1] != 6:
                pose = pose.flatten()[:6]
            
            # Denormalize pose
            pose_mean = self.pose_mean.to(device)
            pose_std = self.pose_std.to(device)
            denormalized_pose = pose * pose_std + pose_mean
            
            translation = denormalized_pose[:3]
            rotation = denormalized_pose[3:6]
            
            # Get category prediction
            category_logits = outputs['category']
            if len(category_logits.shape) > 1:
                category_logits = category_logits.squeeze()
            predicted_category = torch.argmax(category_logits).item()
            category_confidence = torch.softmax(category_logits, dim=0)[predicted_category].item()
            
            # Get confidence
            confidence = outputs['confidence']
            if len(confidence.shape) > 1:
                confidence = confidence.squeeze()
            confidence_value = confidence.item() if confidence.dim() == 0 else confidence.mean().item()
            
            # Get dimensions
            dimensions = outputs['dimensions']
            if len(dimensions.shape) > 1:
                dimensions = dimensions.squeeze()
            
            return {
                'translation': translation.cpu().numpy(),
                'rotation': rotation.cpu().numpy(),
                'predicted_category': predicted_category,
                'confidence': confidence_value,
                'category_confidence': category_confidence,
                'category_name': self.vehicle_categories.get_category_name(predicted_category),
                'dimensions': dimensions.cpu().numpy(),
                'pose_6d': denormalized_pose.cpu().numpy(),
                'attention_weights': outputs['attention'].cpu().numpy(),
                'method': 'vehicle_hspose'
            }
    
    def estimate_distance_and_bearing(self, pose_result):
        """
        Estimate distance and bearing for navigation
        
        Args:
            pose_result (dict): Pose estimation result
        
        Returns:
            dict: Navigation-relevant measurements
        """
        translation = pose_result['translation']
        rotation = pose_result['rotation']
        
        # Calculate distance from ego vehicle
        distance = np.linalg.norm(translation)
        
        # Calculate bearing (angle from forward direction)
        bearing = np.arctan2(translation[1], translation[0])  # Y, X
        
        # Calculate relative heading
        relative_heading = rotation[2]  # Yaw rotation
        
        return {
            'distance': distance,
            'bearing': np.degrees(bearing),
            'relative_heading': np.degrees(relative_heading),
            'lateral_offset': translation[1],
            'longitudinal_distance': translation[0],
            'height_difference': translation[2]
        }

class VehiclePoseLoss(nn.Module):
    """
    Vehicle-specific loss function for pose estimation
    
    Combines multiple loss components optimized for vehicle pose estimation:
    - Pose regression loss with distance-aware weighting
    - Category classification loss with class balancing
    - Confidence estimation loss
    - Dimension estimation loss
    """
    
    def __init__(self, pose_weight=1.0, category_weight=0.5, confidence_weight=0.3, 
                 dimension_weight=0.2, distance_weight=True):
        super(VehiclePoseLoss, self).__init__()
        
        self.pose_weight = pose_weight
        self.category_weight = category_weight
        self.confidence_weight = confidence_weight
        self.dimension_weight = dimension_weight
        self.distance_weight = distance_weight
        
        # Loss functions
        self.pose_loss = nn.MSELoss(reduction='none')
        self.category_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.BCELoss()
        self.dimension_loss = nn.MSELoss()
        
        # Distance-aware weighting for pose loss
        self.pose_weights = torch.tensor([1.0, 1.0, 0.5, 1.0, 1.0, 1.0])  # Lower weight for Z translation
        
    def forward(self, predictions, targets):
        """
        Compute multi-task loss for vehicle pose estimation
        
        Args:
            predictions (dict): Model predictions
            targets (dict): Ground truth targets
        
        Returns:
            dict: Individual and total losses
        """
        # Pose loss with component weighting
        pose_errors = self.pose_loss(predictions['pose'], targets['pose'])
        
        if self.distance_weight:
            # Weight errors based on distance (closer vehicles more important)
            distances = torch.norm(targets['pose'][:, :3], dim=1)
            distance_weights = 1.0 / (1.0 + distances / 50.0)  # Normalize by 50m
            distance_weights = distance_weights.unsqueeze(1).expand(-1, 6)
            pose_errors = pose_errors * distance_weights
        
        # Apply component weights
        device = pose_errors.device
        component_weights = self.pose_weights.to(device).unsqueeze(0)
        weighted_pose_errors = pose_errors * component_weights
        pose_loss = weighted_pose_errors.mean()
        
        # Category classification loss
        category_loss = self.category_loss(predictions['category'], targets['category'])
        
        # Confidence loss
        confidence_loss = self.confidence_loss(predictions['confidence'].squeeze(), targets['confidence'])
        
        # Dimension loss
        dimension_loss = self.dimension_loss(predictions['dimensions'], targets['dimensions'])
        
        # Total weighted loss
        total_loss = (
            self.pose_weight * pose_loss +
            self.category_weight * category_loss +
            self.confidence_weight * confidence_loss +
            self.dimension_weight * dimension_loss
        )
        
        return {
            'total_loss': total_loss,
            'pose_loss': pose_loss,
            'category_loss': category_loss,
            'confidence_loss': confidence_loss,
            'dimension_loss': dimension_loss
        }

# Utility functions for vehicle pose estimation
def load_vehicle_pose_model(checkpoint_path, device='cpu', input_channels=4, num_categories=3):
    """
    Load a trained vehicle pose estimation model
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on
        input_channels (int): Number of input channels
        num_categories (int): Number of vehicle categories
    
    Returns:
        VehiclePoseEstimator: Loaded model
    """
    model = VehiclePoseEstimator(input_channels, num_categories)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded vehicle pose model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded vehicle pose model from {checkpoint_path}")
    else:
        print("Initializing vehicle pose model with random weights")
    
    model.to(device)
    return model

def create_vehicle_pose_optimizer(model, learning_rate=0.001, weight_decay=1e-4):
    """
    Create optimizer for vehicle pose model
    
    Args:
        model (VehiclePoseEstimator): Model to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
    
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Different learning rates for different components
    pose_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'pose_regression' in name:
            pose_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': pose_params, 'lr': learning_rate},
        {'params': other_params, 'lr': learning_rate * 0.1}
    ], weight_decay=weight_decay)
    
    return optimizer

print("🚗 Vehicle-adapted HS-Pose implementation loaded successfully!")
print("🔧 This module provides:")
print("  - VehiclePoseEstimator: Complete vehicle pose estimation model")
print("  - VehiclePoseLoss: Vehicle-optimized loss functions")
print("  - Multi-modal RGB+LIDAR processing")
print("  - Automotive-specific optimizations")
print("📊 Ready for KITTI dataset training and evaluation")