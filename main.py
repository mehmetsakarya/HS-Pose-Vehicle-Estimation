#!/usr/bin/env python3
"""
HS-Pose Vehicle Pose Estimation - Main Execution Script
======================================================

This is the main entry point for the HS-Pose vehicle pose estimation project.
It provides command-line interface for training, evaluation, and demo modes.

Usage:
    python main.py --mode train --dataset ./data/kitti --epochs 50
    python main.py --mode eval --model_path ./models/best_model.pth
    python main.py --mode demo

Author: [Your Name]
Course: Computer Vision
Institution: [Your Institution]
GitHub: https://github.com/[YourUsername]/HS-Pose-Vehicle-Estimation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for HS-Pose vehicle pose estimation"""
    
    # Dataset Configuration
    DATASET_ROOT = "./data/kitti"
    DATASET_TYPE = "kitti"
    
    # Model Configuration
    INPUT_CHANNELS = 4
    NUM_CATEGORIES = 3
    IMAGE_SIZE = (256, 256)
    
    # Training Configuration
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # Loss Weights
    POSE_LOSS_WEIGHT = 1.0
    CATEGORY_LOSS_WEIGHT = 0.5
    CONFIDENCE_LOSS_WEIGHT = 0.3
    
    # Output Configuration
    MODEL_SAVE_PATH = "./models"
    RESULTS_SAVE_PATH = "./results"
    LOG_INTERVAL = 10

# =============================================================================
# VEHICLE CATEGORIES
# =============================================================================

class VehicleCategories:
    """Vehicle category definitions and utilities"""
    
    CATEGORIES = {
        'car': 0,
        'motorcycle': 1,
        'bicycle': 2
    }
    
    CATEGORY_NAMES = {v: k for k, v in CATEGORIES.items()}
    
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
    
    @classmethod
    def get_category_id(cls, category_name):
        return cls.CATEGORIES.get(category_name.lower(), -1)
    
    @classmethod
    def get_category_name(cls, category_id):
        return cls.CATEGORY_NAMES.get(category_id, 'unknown')

# =============================================================================
# HS-POSE MODEL ARCHITECTURE
# =============================================================================

class HybridScopeFeatureExtractor(nn.Module):
    """Hybrid Scope Feature Extractor for vehicle pose estimation"""
    
    def __init__(self, input_channels=4, num_categories=3):
        super(HybridScopeFeatureExtractor, self).__init__()
        
        self.num_categories = num_categories
        
        # Local scope feature extraction
        self.local_conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.local_bn1 = nn.BatchNorm2d(64)
        self.local_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.local_bn2 = nn.BatchNorm2d(128)
        self.local_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.local_bn3 = nn.BatchNorm2d(256)
        
        # Global scope feature extraction
        self.global_conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, padding=3)
        self.global_bn1 = nn.BatchNorm2d(32)
        self.global_conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.global_bn2 = nn.BatchNorm2d(64)
        self.global_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_bn3 = nn.BatchNorm2d(128)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(256 + 128, 512, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(512)
        
        # Pose regression head
        self.pose_regression = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 6)  # 6D pose
        )
        
        # Category classification head
        self.category_classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_categories)
        )
        
        # Confidence estimation head
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
        """Forward pass through the network"""
        # Local scope features
        local_f1 = torch.relu(self.local_bn1(self.local_conv1(x)))
        local_f2 = torch.relu(self.local_bn2(self.local_conv2(local_f1)))
        local_f3 = torch.relu(self.local_bn3(self.local_conv3(local_f2)))
        
        # Global scope features
        global_f1 = torch.relu(self.global_bn1(self.global_conv1(x)))
        global_f2 = torch.relu(self.global_bn2(self.global_conv2(global_f1)))
        global_f3 = torch.relu(self.global_bn3(self.global_conv3(global_f2)))
        
        # Feature fusion
        fused_features = torch.cat([local_f3, global_f3], dim=1)
        fused_features = torch.relu(self.fusion_bn(self.fusion_conv(fused_features)))
        
        # Multi-head predictions
        pose = self.pose_regression(fused_features)
        category = self.category_classifier(fused_features)
        confidence = self.confidence_estimator(fused_features)
        
        return {
            'pose': pose,
            'category': category,
            'confidence': confidence
        }

class VehiclePoseEstimator(nn.Module):
    """Complete vehicle pose estimation model"""
    
    def __init__(self, input_channels=4, num_categories=3):
        super(VehiclePoseEstimator, self).__init__()
        
        self.feature_extractor = HybridScopeFeatureExtractor(input_channels, num_categories)
        self.vehicle_categories = VehicleCategories()
        
    def forward(self, x):
        return self.feature_extractor(x)
    
    def predict_pose(self, image, category_hint=None):
        """Predict 6D pose for a vehicle"""
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                elif len(image.shape) == 4 and image.shape[0] > 1:
                    image = image[:1]
            
            device = next(self.parameters()).device
            image = image.to(device)
            
            outputs = self(image)
            
            # Extract results
            pose = outputs['pose']
            if len(pose.shape) > 1:
                pose = pose.squeeze()
            
            if pose.shape[-1] != 6:
                pose = pose.flatten()[:6]
            
            translation = pose[:3]
            rotation = pose[3:6]
            
            category_logits = outputs['category']
            if len(category_logits.shape) > 1:
                category_logits = category_logits.squeeze()
            predicted_category = torch.argmax(category_logits).item()
            
            confidence = outputs['confidence']
            if len(confidence.shape) > 1:
                confidence = confidence.squeeze()
            confidence_value = confidence.mean().item()
            
            return {
                'translation': translation.cpu().numpy(),
                'rotation': rotation.cpu().numpy(),
                'predicted_category': predicted_category,
                'confidence': confidence_value,
                'category_name': self.vehicle_categories.get_category_name(predicted_category)
            }

# =============================================================================
# KITTI DATASET LOADER
# =============================================================================

class KITTIDatasetLoader:
    """KITTI dataset loader for vehicle pose estimation"""
    
    def __init__(self, kitti_root_path):
        self.kitti_root = kitti_root_path
        self.setup_paths()
        
    def setup_paths(self):
        """Setup KITTI dataset paths"""
        self.paths = {
            'images': os.path.join(self.kitti_root, 'training/image_2'),
            'labels': os.path.join(self.kitti_root, 'training/label_2'),
            'lidar': os.path.join(self.kitti_root, 'training/velodyne'),
            'calib': os.path.join(self.kitti_root, 'training/calib')
        }
        
        print("KITTI Dataset Paths:")
        for key, path in self.paths.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {key}: {path} {exists}")
    
    def create_synthetic_data(self, num_samples=50):
        """Create synthetic data for testing"""
        print(f"Creating {num_samples} synthetic KITTI samples...")
        
        samples = []
        vehicle_types = ['Car', 'Van', 'Truck']
        
        for i in range(num_samples):
            vehicle_type = np.random.choice(vehicle_types)
            
            vehicles = [{
                'type': vehicle_type,
                'truncated': np.random.uniform(0.0, 0.3),
                'occluded': np.random.randint(0, 3),
                'alpha': np.random.uniform(-np.pi, np.pi),
                'bbox': [
                    np.random.uniform(100, 400),
                    np.random.uniform(100, 200),
                    np.random.uniform(500, 800),
                    np.random.uniform(300, 400)
                ],
                'dimensions': [1.5, 1.8, 4.2],  # h, w, l
                'location': [
                    np.random.uniform(-10, 10),
                    np.random.uniform(-2, 2),
                    np.random.uniform(5, 50)
                ],
                'rotation_y': np.random.uniform(-np.pi, np.pi)
            }]
            
            sample = {
                'file_id': f"synthetic_{i:06d}",
                'vehicles': vehicles,
                'synthetic': True
            }
            samples.append(sample)
        
        return samples

# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class VehiclePoseTrainer:
    """Training pipeline for vehicle pose estimation"""
    
    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config or Config()
        
        self.model.to(device)
        
        self.pose_loss_fn = nn.MSELoss()
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, outputs, targets):
        """Compute multi-task loss"""
        pred_pose = outputs['pose']
        pred_category = outputs['category']
        pred_confidence = outputs['confidence']
        
        gt_pose = targets['pose']
        gt_category = targets['category']
        gt_confidence = targets['confidence']
        
        # Reshape if necessary
        if len(pred_pose.shape) > 2:
            pred_pose = pred_pose.view(pred_pose.shape[0], -1)
        if len(pred_category.shape) > 2:
            pred_category = pred_category.view(pred_category.shape[0], -1)
        if len(pred_confidence.shape) > 2:
            pred_confidence = pred_confidence.view(pred_confidence.shape[0], -1).mean(dim=1)
        
        # Ensure pose dimensions match
        if pred_pose.shape[1] != gt_pose.shape[1]:
            if pred_pose.shape[1] > gt_pose.shape[1]:
                pred_pose = pred_pose[:, :gt_pose.shape[1]]
            else:
                padding = torch.zeros(pred_pose.shape[0], gt_pose.shape[1] - pred_pose.shape[1]).to(self.device)
                pred_pose = torch.cat([pred_pose, padding], dim=1)
        
        # Compute losses
        pose_loss = self.pose_loss_fn(pred_pose, gt_pose)
        category_loss = self.category_loss_fn(pred_category, gt_category)
        confidence_loss = self.confidence_loss_fn(pred_confidence, gt_confidence)
        
        total_loss = (self.config.POSE_LOSS_WEIGHT * pose_loss + 
                     self.config.CATEGORY_LOSS_WEIGHT * category_loss + 
                     self.config.CONFIDENCE_LOSS_WEIGHT * confidence_loss)
        
        return {
            'total_loss': total_loss,
            'pose_loss': pose_loss,
            'category_loss': category_loss,
            'confidence_loss': confidence_loss
        }
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        self.model.train()
        
        images = batch['images'].to(self.device)
        targets = {
            'pose': batch['poses'].to(self.device),
            'category': batch['categories'].to(self.device),
            'confidence': batch['confidences'].to(self.device)
        }
        
        optimizer.zero_grad()
        
        outputs = self.model(images)
        losses = self.compute_loss(outputs, targets)
        
        losses['total_loss'].backward()
        optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_training_data(config):
    """Prepare training data"""
    print("🔄 Preparing training data...")
    
    kitti_loader = KITTIDatasetLoader(config.DATASET_ROOT)
    samples = kitti_loader.create_synthetic_data(num_samples=100)
    
    training_data = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            # Create synthetic image and depth
            image = np.random.rand(*config.IMAGE_SIZE, 3).astype(np.float32)
            depth_map = np.random.rand(*config.IMAGE_SIZE).astype(np.float32)
            
            for vehicle in sample['vehicles']:
                # Extract pose
                location = np.array(vehicle['location'])
                rotation = np.array([0, 0, vehicle['rotation_y']])
                pose_6d = np.concatenate([location, rotation])
                
                # Map category
                category_map = {'Car': 0, 'Van': 0, 'Truck': 0}
                category = category_map.get(vehicle['type'], 0)
                
                # Create 4-channel input
                depth_resized = np.expand_dims(depth_map, axis=-1)
                input_data = np.concatenate([image, depth_resized], axis=-1)
                
                training_sample = {
                    'image': input_data,
                    'pose': pose_6d,
                    'category': category,
                    'confidence': 1.0
                }
                
                training_data.append(training_sample)
        
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"✅ Prepared {len(training_data)} training samples")
    return training_data

def create_data_loader(training_data, batch_size, shuffle=True):
    """Create PyTorch data loader"""
    from torch.utils.data import Dataset, DataLoader
    
    class VehicleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            
            image = torch.FloatTensor(sample['image']).permute(2, 0, 1)
            pose = torch.FloatTensor(sample['pose'])
            category = torch.LongTensor([sample['category']])
            confidence = torch.FloatTensor([sample['confidence']])
            
            return {
                'images': image,
                'poses': pose,
                'categories': category.squeeze(),
                'confidences': confidence.squeeze()
            }
    
    dataset = VehicleDataset(training_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def setup_environment():
    """Setup environment"""
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_SAVE_PATH, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device

def train_model(config):
    """Main training function"""
    print("🚀 Starting HS-Pose Vehicle Training...")
    print("=" * 50)
    
    device = setup_environment()
    
    # Prepare data
    training_data = prepare_training_data(config)
    
    if len(training_data) == 0:
        print("❌ No training data available!")
        return None
    
    # Split data
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Create data loaders
    train_loader = create_data_loader(train_data, config.BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_data, 1, shuffle=False)
    
    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Validation samples: {len(val_data)}")
    
    # Initialize model and trainer
    model = VehiclePoseEstimator(config.INPUT_CHANNELS, config.NUM_CATEGORIES)
    trainer = VehiclePoseTrainer(model, device, config)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"🎯 Starting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in train_pbar:
            losses = trainer.train_step(batch, optimizer)
            epoch_losses.append(losses['total_loss'])
            
            train_pbar.set_postfix({
                'Loss': f"{losses['total_loss']:.4f}",
                'Pose': f"{losses['pose_loss']:.4f}",
                'Cat': f"{losses['category_loss']:.4f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                targets = {
                    'pose': batch['poses'].to(device),
                    'category': batch['categories'].to(device),
                    'confidence': batch['confidences'].to(device)
                }
                
                outputs = model(images)
                losses = trainer.compute_loss(outputs, targets)
                val_losses.append(losses['total_loss'].item())
        
        scheduler.step()
        
        avg_train_loss = np.mean(epoch_losses)
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, model_path)
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
    
    print("✅ Training completed!")
    return model

def evaluate_model(model_path, config):
    """Evaluate trained model"""
    device = setup_environment()
    
    print("📊 Evaluating model...")
    
    # Load model
    model = VehiclePoseEstimator(config.INPUT_CHANNELS, config.NUM_CATEGORIES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare test data
    test_data = prepare_training_data(config)
    test_loader = create_data_loader(test_data[:30], 1, shuffle=False)
    
    # Evaluation
    translation_errors = []
    rotation_errors = []
    category_accuracies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['images'].to(device)
            gt_poses = batch['poses'].to(device)
            gt_categories = batch['categories'].to(device)
            
            outputs = model(images)
            pred_poses = outputs['pose']
            pred_categories = torch.argmax(outputs['category'], dim=1)
            
            trans_error = torch.norm(pred_poses[:, :3] - gt_poses[:, :3], dim=1)
            rot_error = torch.norm(pred_poses[:, 3:6] - gt_poses[:, 3:6], dim=1)
            cat_accuracy = (pred_categories == gt_categories).float()
            
            translation_errors.extend(trans_error.cpu().numpy())
            rotation_errors.extend(rot_error.cpu().numpy())
            category_accuracies.extend(cat_accuracy.cpu().numpy())
    
    # Results
    results = {
        'mean_translation_error': np.mean(translation_errors),
        'std_translation_error': np.std(translation_errors),
        'mean_rotation_error': np.mean(rotation_errors),
        'std_rotation_error': np.std(rotation_errors),
        'category_accuracy': np.mean(category_accuracies)
    }
    
    print(f"📈 Evaluation Results:")
    print(f"   Translation Error: {results['mean_translation_error']:.3f} ± {results['std_translation_error']:.3f} m")
    print(f"   Rotation Error: {results['mean_rotation_error']:.3f} ± {results['std_rotation_error']:.3f} rad")
    print(f"   Category Accuracy: {results['category_accuracy']:.1%}")
    
    return results

def demo_inference():
    """Demo inference"""
    print("🎯 Demo Mode - Quick Inference Test")
    
    device = setup_environment()
    model = VehiclePoseEstimator(4, 3)
    model.to(device)
    
    # Create sample input
    sample_input = torch.randn(1, 4, 256, 256).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(sample_input)
        print(f"✅ Demo inference successful!")
        print(f"   Pose shape: {outputs['pose'].shape}")
        print(f"   Category shape: {outputs['category'].shape}")
        print(f"   Confidence shape: {outputs['confidence'].shape}")
        
        # Get prediction result
        result = model.predict_pose(sample_input, "car")
        print(f"📊 Sample prediction:")
        print(f"   Category: {result['category_name']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Translation: {result['translation']}")
        print(f"   Rotation: {result['rotation']}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='HS-Pose Vehicle Pose Estimation')
    parser.add_argument('--mode', choices=['train', 'eval', 'demo'], default='demo',
                       help='Execution mode')
    parser.add_argument('--dataset', default='./data/kitti',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_path', default='./models/best_model.pth',
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.DATASET_ROOT = args.dataset
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    if args.mode == 'train':
        model = train_model(config)
        if model is not None:
            print("🎉 Training completed successfully!")
    
    elif args.mode == 'eval':
        if os.path.exists(args.model_path):
            results = evaluate_model(args.model_path, config)
        else:
            print(f"❌ Model file not found: {args.model_path}")
    
    elif args.mode == 'demo':
        demo_inference()

if __name__ == "__main__":
    main()
