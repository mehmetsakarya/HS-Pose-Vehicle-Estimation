"""
KITTI Dataset Loader for Vehicle Pose Estimation
===============================================

This module provides comprehensive functionality for loading and preprocessing
KITTI dataset including images, LIDAR point clouds, calibration data, and labels
for vehicle pose estimation tasks.

The KITTI dataset is one of the most widely used datasets for autonomous driving
research, providing synchronized camera images, LIDAR point clouds, and GPS/IMU data.

Key Features:
- KITTI 3D object detection dataset support
- LIDAR point cloud to depth image conversion
- Camera-LIDAR calibration handling
- Vehicle annotation parsing
- Data preprocessing and augmentation
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

class KITTIDatasetLoader:
    """
    Comprehensive KITTI dataset loader for vehicle pose estimation
    
    This class provides functionality for loading and preprocessing KITTI dataset
    including images, LIDAR point clouds, calibration data, and vehicle labels.
    """
    
    def __init__(self, kitti_root_path: str, split: str = "training"):
        """
        Initialize KITTI dataset loader
        
        Args:
            kitti_root_path (str): Path to KITTI dataset root directory
            split (str): Dataset split ('training' or 'testing')
        """
        self.kitti_root = Path(kitti_root_path)
        self.split = split
        self.setup_paths()
        
        # Vehicle type mapping for KITTI
        self.vehicle_type_mapping = {
            'Car': 0,
            'Van': 0,
            'Truck': 0,
            'Pedestrian': -1,  # Not a vehicle
            'Person_sitting': -1,  # Not a vehicle
            'Cyclist': 2,  # Map to bicycle
            'Tram': 0,  # Map to car
            'Misc': -1,  # Unknown
            'DontCare': -1  # Ignore
        }
        
        # KITTI label field definitions
        self.label_fields = [
            'type', 'truncated', 'occluded', 'alpha',
            'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
            'height', 'width', 'length',
            'location_x', 'location_y', 'location_z',
            'rotation_y'
        ]
        
    def setup_paths(self):
        """Setup KITTI dataset directory paths"""
        self.paths = {
            'images': self.kitti_root / self.split / 'image_2',
            'labels': self.kitti_root / self.split / 'label_2',
            'lidar': self.kitti_root / self.split / 'velodyne',
            'calib': self.kitti_root / self.split / 'calib'
        }
        
        print("KITTI Dataset Paths:")
        for key, path in self.paths.items():
            exists = "✓" if path.exists() else "✗"
            print(f"  {key}: {path} {exists}")
            
        # Check if this is a valid KITTI dataset
        self.is_valid_dataset = all(path.exists() for path in self.paths.values())
        
        if not self.is_valid_dataset:
            print("⚠️ Warning: Some KITTI dataset directories are missing")
            print("   This loader will work with synthetic data for demonstration")
    
    def parse_kitti_label(self, label_file: Union[str, Path]) -> List[Dict]:
        """
        Parse KITTI label file to extract object annotations
        
        Args:
            label_file (str or Path): Path to KITTI label file
        
        Returns:
            list: List of object annotations
        """
        objects = []
        label_path = Path(label_file)
        
        if not label_path.exists():
            return objects
            
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f.readlines()):
                    parts = line.strip().split(' ')
                    if len(parts) >= 15:
                        try:
                            obj = {
                                'type': parts[0],
                                'truncated': float(parts[1]),
                                'occluded': int(parts[2]),
                                'alpha': float(parts[3]),
                                'bbox': [float(x) for x in parts[4:8]],  # [left, top, right, bottom]
                                'dimensions': [float(x) for x in parts[8:11]],  # [height, width, length]
                                'location': [float(x) for x in parts[11:14]],  # [x, y, z] in camera coordinates
                                'rotation_y': float(parts[14]),
                                'score': float(parts[15]) if len(parts) > 15 else 1.0
                            }
                            objects.append(obj)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Error parsing line {line_num + 1} in {label_file}: {e}")
                            continue
        except Exception as e:
            print(f"Error reading label file {label_file}: {e}")
            
        return objects
    
    def load_calib_file(self, calib_file: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load KITTI calibration file
        
        Args:
            calib_file (str or Path): Path to calibration file
        
        Returns:
            dict: Calibration parameters as numpy arrays
        """
        calib = {}
        calib_path = Path(calib_file)
        
        if not calib_path.exists():
            print(f"Calibration file not found: {calib_file}, using default")
            return self.get_default_calib()
        
        try:
            with open(calib_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if ':' in line and not line.startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse numerical values
                        try:
                            numbers = [float(x) for x in value.split()]
                            
                            # Reshape based on known calibration matrices
                            if key in ['P0', 'P1', 'P2', 'P3']:
                                calib[key] = np.array(numbers).reshape(3, 4)
                            elif key == 'R0_rect':
                                calib[key] = np.array(numbers).reshape(3, 3)
                            elif key == 'Tr_velo_to_cam':
                                calib[key] = np.array(numbers).reshape(3, 4)
                            else:
                                calib[key] = np.array(numbers)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not parse calibration line: {line}")
                            continue
        except Exception as e:
            print(f"Error reading calibration file {calib_file}: {e}")
            return self.get_default_calib()
        
        return calib
    
    def get_default_calib(self) -> Dict[str, np.ndarray]:
        """
        Get default KITTI calibration parameters
        
        Returns:
            dict: Default calibration matrices
        """
        return {
            'P2': np.array([
                [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
            ]),
            'R0_rect': np.array([
                [9.999239e-01, 9.837760e-03, -7.445048e-03],
                [-9.869795e-03, 9.999421e-01, -4.278459e-03],
                [7.402527e-03, 4.351614e-03, 9.999631e-01]
            ]),
            'Tr_velo_to_cam': np.array([
                [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]
            ])
        }
    
    def load_lidar_data(self, lidar_file: Union[str, Path]) -> np.ndarray:
        """
        Load LIDAR point cloud data from KITTI binary format
        
        Args:
            lidar_file (str or Path): Path to LIDAR file (.bin format)
        
        Returns:
            np.ndarray: LIDAR points array (N, 4) for x, y, z, intensity
        """
        lidar_path = Path(lidar_file)
        
        if not lidar_path.exists():
            raise FileNotFoundError(f"LIDAR file not found: {lidar_file}")
        
        if lidar_path.suffix == '.bin':
            # KITTI binary format
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            return points  # x, y, z, intensity
        else:
            raise ValueError(f"Unsupported LIDAR file format: {lidar_path.suffix}")
    
    def project_lidar_to_image(self, points: np.ndarray, calib: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project LIDAR points to image coordinates using calibration matrices
        
        Args:
            points (np.ndarray): LIDAR points (N, 3 or 4)
            calib (dict): Calibration parameters
        
        Returns:
            tuple: (projected_points (N, 2), valid_indices (N,))
        """
        # Take only x, y, z coordinates
        if points.shape[1] > 3:
            points_xyz = points[:, :3]
        else:
            points_xyz = points
        
        # Convert to homogeneous coordinates
        points_hom = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        
        # Transform from Velodyne to camera coordinates
        # Step 1: Velodyne to unrectified camera
        Tr_velo_to_cam = calib.get('Tr_velo_to_cam', self.get_default_calib()['Tr_velo_to_cam'])
        Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        points_cam_unrect = np.dot(Tr_velo_to_cam_hom, points_hom.T).T
        
        # Step 2: Apply rectification
        R0_rect = calib.get('R0_rect', self.get_default_calib()['R0_rect'])
        R0_rect_hom = np.hstack([R0_rect, np.zeros((3, 1))])
        R0_rect_hom = np.vstack([R0_rect_hom, [0, 0, 0, 1]])
        points_cam_rect = np.dot(R0_rect_hom, points_cam_unrect.T).T
        
        # Remove points behind camera (z <= 0)
        valid_idx = points_cam_rect[:, 2] > 0
        points_cam_rect = points_cam_rect[valid_idx]
        
        if len(points_cam_rect) == 0:
            return np.array([]).reshape(0, 2), valid_idx
        
        # Step 3: Project to image coordinates using camera matrix
        P2 = calib.get('P2', self.get_default_calib()['P2'])
        points_img_hom = np.dot(P2, points_cam_rect[:, :3].T).T
        
        # Convert from homogeneous to 2D coordinates
        points_img = points_img_hom[:, :2] / points_img_hom[:, 2:3]
        
        return points_img, valid_idx
    
    def create_depth_map_from_lidar(self, lidar_file: Union[str, Path], 
                                  calib_file: Union[str, Path], 
                                  image_shape: Tuple[int, int] = (375, 1242)) -> np.ndarray:
        """
        Create depth map from LIDAR point cloud
        
        Args:
            lidar_file (str or Path): Path to LIDAR file
            calib_file (str or Path): Path to calibration file
            image_shape (tuple): Target image shape (height, width)
        
        Returns:
            np.ndarray: Depth map image
        """
        try:
            # Load LIDAR points and calibration
            points = self.load_lidar_data(lidar_file)
            calib = self.load_calib_file(calib_file)
            
            # Project LIDAR points to image coordinates
            points_2d, valid_idx = self.project_lidar_to_image(points, calib)
            
            # Initialize depth map
            depth_map = np.zeros(image_shape, dtype=np.float32)
            
            if len(points_2d) > 0:
                # Filter points within image bounds
                height, width = image_shape
                valid_2d = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & 
                           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height))
                
                if np.any(valid_2d):
                    x_coords = points_2d[valid_2d, 0].astype(int)
                    y_coords = points_2d[valid_2d, 1].astype(int)
                    
                    # Get depth values (z coordinate in camera frame)
                    valid_points = points[valid_idx][valid_2d]
                    depths = valid_points[:, 2]  # Z coordinate as depth
                    
                    # Handle multiple points projecting to the same pixel (take closest)
                    for i in range(len(x_coords)):
                        x, y, d = x_coords[i], y_coords[i], depths[i]
                        if depth_map[y, x] == 0 or d < depth_map[y, x]:
                            depth_map[y, x] = d
            
            return depth_map
            
        except Exception as e:
            print(f"Error creating depth map: {e}")
            # Return empty depth map in case of error
            return np.zeros(image_shape, dtype=np.float32)
    
    def get_vehicle_samples(self, max_samples: int = 100) -> List[Dict]:
        """
        Extract vehicle samples from KITTI dataset
        
        Args:
            max_samples (int): Maximum number of samples to extract
        
        Returns:
            list: List of vehicle samples with annotations
        """
        samples = []
        vehicle_types = ['Car', 'Van', 'Truck', 'Cyclist']
        
        # Check if this is a valid KITTI dataset
        if not self.is_valid_dataset:
            print("KITTI dataset not found. Creating synthetic sample data...")
            return self.create_synthetic_data(max_samples)
        
        # Get all label files
        label_files = sorted(list(self.paths['labels'].glob('*.txt')))
        
        print(f"Processing KITTI {self.split} split...")
        
        for label_file in tqdm(label_files[:max_samples], desc="Loading KITTI samples"):
            file_id = label_file.stem
            
            #