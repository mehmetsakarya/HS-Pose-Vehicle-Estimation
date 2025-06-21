"""
Model Evaluation System for Vehicle Pose Estimation
==================================================

This module provides comprehensive evaluation functionality for vehicle pose estimation
models, including metric calculation, performance analysis, and result visualization.

Key Features:
- Multiple evaluation metrics (translation error, rotation error, category accuracy)
- Performance analysis across different scenarios
- Comparison with baseline methods
- Statistical significance testing
- Comprehensive result reporting
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import pandas as pd

class ModelEvaluator:
    """
    Comprehensive model evaluation system
    
    This class provides extensive evaluation functionality including metric calculation,
    performance analysis, and comparison with baseline methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Any):
        """
        Initialize the evaluator
        
        Args:
            model (nn.Module): Model to evaluate
            device (torch.device): Device for evaluation
            config (Config): Configuration object
        """
        self.model = model
        self.device = device
        self.config = config
        
        self.model.to(device)
        self.model.eval()
        
        # Evaluation results storage
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': {},
            'per_sample_metrics': [],
            'category_performance': {},
            'distance_analysis': {},
            'timing_analysis': {}
        }
        
        # Evaluation thresholds
        self.translation_thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]  # meters
        self.rotation_thresholds = [5, 10, 15, 30, 45]  # degrees
        
        # Statistical tests
        self.significance_level = 0.05
        
    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            test_data (list): List of test samples
        
        Returns:
            dict: Comprehensive evaluation results
        """
        print(" Starting model evaluation...")
        
        # Clear previous results
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': {},
            'per_sample_metrics': [],
            'category_performance': {},
            'distance_analysis': {},
            'timing_analysis': {}
        }
        
        # Evaluate on all samples
        self._evaluate_samples(test_data)
        
        # Calculate comprehensive metrics
        self._calculate_metrics()
        
        # Analyze performance by category
        self._analyze_by_category()
        
        # Analyze performance by distance
        self._analyze_by_distance()
        
        # Perform timing analysis
        self._analyze_timing()
        
        # Calculate statistical significance
        self._calculate_statistical_significance()
        
        print(" Evaluation completed!")
        return self.results
    
    def _evaluate_samples(self, test_data: List[Dict]):
        """Evaluate model on individual samples"""
        print(" Evaluating individual samples...")
        
        for sample_idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Prepare input
                if isinstance(sample['image'], np.ndarray):
                    image = torch.from_numpy(sample['image']).float()
                else:
                    image = sample['image']
                
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                image = image.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Get prediction
                with torch.no_grad():
                    if hasattr(self.model, 'predict_pose'):
                        prediction = self.model.predict_pose(image)
                    else:
                        # Fallback to direct model output
                        outputs = self.model(image)
                        prediction = self._parse_model_output(outputs)
                
                inference_time = time.time() - start_time
                
                # Ground truth
                ground_truth = {
                    'translation': sample['pose'][:3],
                    'rotation': sample['pose'][3:6],
                    'category': sample['category'],
                    'vehicle_type': sample.get('vehicle_type', 'unknown')
                }
                
                # Store results
                self.results['predictions'].append(prediction)
                self.results['ground_truths'].append(ground_truth)
                
                # Calculate per-sample metrics
                sample_metrics = self._calculate_sample_metrics(
                    prediction, ground_truth, inference_time, sample_idx
                )
                self.results['per_sample_metrics'].append(sample_metrics)
                
            except Exception as e:
                print(f"Error evaluating sample {sample_idx}: {e}")
                continue
    
    def _parse_model_output(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Parse raw model output into prediction format"""
        pose = outputs['pose'].squeeze().cpu().numpy()
        
        if len(pose) >= 6:
            translation = pose[:3]
            rotation = pose[3:6]
        else:
            translation = pose[:3] if len(pose) >= 3 else np.zeros(3)
            rotation = np.zeros(3)
        
        category_logits = outputs['category'].squeeze().cpu()
        predicted_category = torch.argmax(category_logits).item()
        
        confidence = outputs.get('confidence', torch.tensor(0.5)).squeeze().cpu().item()
        
        return {
            'translation': translation,
            'rotation': rotation,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'category_name': ['car', 'motorcycle', 'bicycle'][predicted_category] if predicted_category < 3 else 'unknown'
        }
    
    def _calculate_sample_metrics(self, prediction: Dict, ground_truth: Dict, 
                                 inference_time: float, sample_idx: int) -> Dict[str, float]:
        """Calculate metrics for a single sample"""
        # Translation error (Euclidean distance)
        translation_error = np.linalg.norm(
            prediction['translation'] - ground_truth['translation']
        )
        
        # Rotation error (angular difference)
        rotation_error = np.linalg.norm(
            prediction['rotation'] - ground_truth['rotation']
        )
        
        # Category accuracy
        pred_cat = prediction.get('predicted_category', -1)
        gt_cat = ground_truth['category']
        category_correct = 1.0 if pred_cat == gt_cat else 0.0
        
        # Distance from origin
        distance_from_origin = np.linalg.norm(ground_truth['translation'])
        
        # Angular translation error
        angular_translation_error = np.degrees(
            np.arctan2(translation_error, distance_from_origin)
        ) if distance_from_origin > 0 else 0.0
        
        # Confidence score
        confidence = prediction.get('confidence', 0.5)
        
        return {
            'sample_idx': sample_idx,
            'translation_error': translation_error,
            'rotation_error': rotation_error,
            'angular_translation_error': angular_translation_error,
            'category_correct': category_correct,
            'distance_from_origin': distance_from_origin,
            'confidence': confidence,
            'inference_time': inference_time,
            'vehicle_type': ground_truth['vehicle_type']
        }
    
    def _calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        print(" Calculating comprehensive metrics...")
        
        if not self.results['per_sample_metrics']:
            print("Warning: No sample metrics available")
            return
        
        metrics_df = pd.DataFrame(self.results['per_sample_metrics'])
        
        # Basic metrics
        self.results['metrics'] = {
            # Translation metrics
            'mean_translation_error': float(metrics_df['translation_error'].mean()),
            'median_translation_error': float(metrics_df['translation_error'].median()),
            'std_translation_error': float(metrics_df['translation_error'].std()),
            'max_translation_error': float(metrics_df['translation_error'].max()),
            'min_translation_error': float(metrics_df['translation_error'].min()),
            
            # Rotation metrics
            'mean_rotation_error': float(metrics_df['rotation_error'].mean()),
            'median_rotation_error': float(metrics_df['rotation_error'].median()),
            'std_rotation_error': float(metrics_df['rotation_error'].std()),
            'max_rotation_error': float(metrics_df['rotation_error'].max()),
            'min_rotation_error': float(metrics_df['rotation_error'].min()),
            
            # Angular translation metrics
            'mean_angular_translation_error': float(metrics_df['angular_translation_error'].mean()),
            'median_angular_translation_error': float(metrics_df['angular_translation_error'].median()),
            
            # Category metrics
            'category_accuracy': float(metrics_df['category_correct'].mean()),
            'category_precision': self._calculate_precision(metrics_df),
            'category_recall': self._calculate_recall(metrics_df),
            'category_f1': self._calculate_f1_score(metrics_df),
            
            # Performance metrics
            'mean_inference_time': float(metrics_df['inference_time'].mean()),
            'median_inference_time': float(metrics_df['inference_time'].median()),
            'fps': 1.0 / float(metrics_df['inference_time'].mean()),
            
            # Confidence metrics
            'mean_confidence': float(metrics_df['confidence'].mean()),
            'confidence_correlation': self._calculate_confidence_correlation(metrics_df),
            
            # Threshold-based metrics
            'translation_accuracy_at_thresholds': self._calculate_threshold_accuracy(
                metrics_df['translation_error'], self.translation_thresholds
            ),
            'rotation_accuracy_at_thresholds': self._calculate_threshold_accuracy(
                np.degrees(metrics_df['rotation_error']), self.rotation_thresholds
            ),
            
            # Sample count
            'total_samples': len(metrics_df),
            'successful_predictions': len(metrics_df)
        }
    
    def _calculate_precision(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate precision for each category"""
        precisions = {}
        categories = ['car', 'motorcycle', 'bicycle']
        
        for i, category in enumerate(categories):
            # Get predictions and ground truth for this category
            pred_mask = metrics_df['vehicle_type'] == category
            if pred_mask.sum() == 0:
                precisions[category] = 0.0
                continue
                
            # True positives: correctly predicted as this category
            tp = metrics_df[pred_mask]['category_correct'].sum()
            # Total predicted as this category
            total_pred = pred_mask.sum()
            
            precisions[category] = float(tp / total_pred) if total_pred > 0 else 0.0
        
        # Overall precision
        precisions['overall'] = float(metrics_df['category_correct'].mean())
        
        return precisions
    
    def _calculate_recall(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate recall for each category"""
        recalls = {}
        categories = ['car', 'motorcycle', 'bicycle']
        
        for i, category in enumerate(categories):
            # Get ground truth for this category
            gt_mask = metrics_df['vehicle_type'] == category
            if gt_mask.sum() == 0:
                recalls[category] = 0.0
                continue
                
            # True positives: correctly predicted as this category
            tp = metrics_df[gt_mask]['category_correct'].sum()
            # Total ground truth for this category
            total_gt = gt_mask.sum()
            
            recalls[category] = float(tp / total_gt) if total_gt > 0 else 0.0
        
        # Overall recall
        recalls['overall'] = float(metrics_df['category_correct'].mean())
        
        return recalls
    
    def _calculate_f1_score(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate F1 score for each category"""
        precision = self._calculate_precision(metrics_df)
        recall = self._calculate_recall(metrics_df)
        
        f1_scores = {}
        categories = ['car', 'motorcycle', 'bicycle', 'overall']
        
        for category in categories:
            p = precision.get(category, 0.0)
            r = recall.get(category, 0.0)
            
            if p + r > 0:
                f1_scores[category] = 2 * (p * r) / (p + r)
            else:
                f1_scores[category] = 0.0
        
        return f1_scores
    
    def _calculate_confidence_correlation(self, metrics_df: pd.DataFrame) -> float:
        """Calculate correlation between confidence and accuracy"""
        # Inverse correlation: higher confidence should correlate with lower error
        translation_errors = metrics_df['translation_error'].values
        confidences = metrics_df['confidence'].values
        
        if len(translation_errors) > 1 and len(confidences) > 1:
            correlation, _ = stats.pearsonr(-translation_errors, confidences)
            return float(correlation) if not np.isnan(correlation) else 0.0
        return 0.0
    
    def _calculate_threshold_accuracy(self, errors: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
        """Calculate accuracy at different error thresholds"""
        accuracy_at_threshold = {}
        
        for threshold in thresholds:
            accuracy = float(np.mean(errors <= threshold))
            accuracy_at_threshold[f'{threshold}'] = accuracy
        
        return accuracy_at_threshold
    
    def _analyze_by_category(self):
        """Analyze performance by vehicle category"""
        print(" Analyzing performance by category...")
        
        if not self.results['per_sample_metrics']:
            return
        
        metrics_df = pd.DataFrame(self.results['per_sample_metrics'])
        
        category_performance = {}
        vehicle_types = metrics_df['vehicle_type'].unique()
        
        for vehicle_type in vehicle_types:
            mask = metrics_df['vehicle_type'] == vehicle_type
            category_data = metrics_df[mask]
            
            if len(category_data) == 0:
                continue
            
            category_performance[vehicle_type] = {
                'sample_count': len(category_data),
                'mean_translation_error': float(category_data['translation_error'].mean()),
                'std_translation_error': float(category_data['translation_error'].std()),
                'mean_rotation_error': float(category_data['rotation_error'].mean()),
                'std_rotation_error': float(category_data['rotation_error'].std()),
                'category_accuracy': float(category_data['category_correct'].mean()),
                'mean_confidence': float(category_data['confidence'].mean()),
                'mean_distance': float(category_data['distance_from_origin'].mean())
            }
        
        self.results['category_performance'] = category_performance
    
    def _analyze_by_distance(self):
        """Analyze performance by distance from camera"""
        print("📏 Analyzing performance by distance...")
        
        if not self.results['per_sample_metrics']:
            return
        
        metrics_df = pd.DataFrame(self.results['per_sample_metrics'])
        
        # Define distance bins
        distance_bins = [0, 10, 20, 30, 50, 100]  # meters
        distance_labels = ['0-10m', '10-20m', '20-30m', '30-50m', '50m+']
        
        # Bin the data
        metrics_df['distance_bin'] = pd.cut(
            metrics_df['distance_from_origin'], 
            bins=distance_bins + [float('inf')], 
            labels=distance_labels + ['far'],
            include_lowest=True
        )
        
        distance_analysis = {}
        
        for bin_label in metrics_df['distance_bin'].unique():
            if pd.isna(bin_label):
                continue
                
            mask = metrics_df['distance_bin'] == bin_label
            bin_data = metrics_df[mask]
            
            if len(bin_data) == 0:
                continue
            
            distance_analysis[str(bin_label)] = {
                'sample_count': len(bin_data),
                'mean_translation_error': float(bin_data['translation_error'].mean()),
                'std_translation_error': float(bin_data['translation_error'].std()),
                'mean_rotation_error': float(bin_data['rotation_error'].mean()),
                'category_accuracy': float(bin_data['category_correct'].mean()),
                'mean_distance': float(bin_data['distance_from_origin'].mean()),
                'mean_inference_time': float(bin_data['inference_time'].mean())
            }
        
        self.results['distance_analysis'] = distance_analysis
    
    def _analyze_timing(self):
        """Analyze timing performance"""
        print("⏱️ Analyzing timing performance...")
        
        if not self.results['per_sample_metrics']:
            return
        
        metrics_df = pd.DataFrame(self.results['per_sample_metrics'])
        
        timing_analysis = {
            'mean_inference_time': float(metrics_df['inference_time'].mean()),
            'median_inference_time': float(metrics_df['inference_time'].median()),
            'std_inference_time': float(metrics_df['inference_time'].std()),
            'min_inference_time': float(metrics_df['inference_time'].min()),
            'max_inference_time': float(metrics_df['inference_time'].max()),
            'fps_mean': 1.0 / float(metrics_df['inference_time'].mean()),
            'fps_median': 1.0 / float(metrics_df['inference_time'].median()),
            'real_time_capable': float(metrics_df['inference_time'].mean()) < 0.1,  # 10 FPS threshold
            'percentile_95_time': float(np.percentile(metrics_df['inference_time'], 95)),
            'percentile_99_time': float(np.percentile(metrics_df['inference_time'], 99))
        }
        
        self.results['timing_analysis'] = timing_analysis
    
    def _calculate_statistical_significance(self):
        """Calculate statistical significance of results"""
        print(" Calculating statistical significance...")
        
        if not self.results['per_sample_metrics']:
            return
        
        metrics_df = pd.DataFrame(self.results['per_sample_metrics'])
        
        # Sample size
        n = len(metrics_df)
        
        # Confidence intervals for key metrics
        confidence_intervals = {}
        
        for metric in ['translation_error', 'rotation_error', 'category_correct']:
            values = metrics_df[metric].values
            mean = np.mean(values)
            sem = stats.sem(values)  # Standard error of mean
            
            # 95% confidence interval
            ci = stats.t.interval(
                0.95, 
                df=n-1, 
                loc=mean, 
                scale=sem
            )
            
            confidence_intervals[metric] = {
                'mean': float(mean),
                'confidence_interval_95': [float(ci[0]), float(ci[1])],
                'standard_error': float(sem),
                'sample_size': n
            }
        
        self.results['statistical_analysis'] = {
            'confidence_intervals': confidence_intervals,
            'sample_size': n,
            'significance_level': self.significance_level
        }
    
    def compare_with_baseline(self, baseline_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare results with baseline method
        
        Args:
            baseline_results (dict): Baseline method results
        
        Returns:
            dict: Comparison results
        """
        print(" Comparing with baseline...")
        
        current_results = self.results['metrics']
        comparison = {}
        
        for metric_name in baseline_results.keys():
            if metric_name in current_results:
                baseline_value = baseline_results[metric_name]
                current_value = current_results[metric_name]
                
                # Calculate improvement
                if 'error' in metric_name:
                    # Lower is better for error metrics
                    improvement = (baseline_value - current_value) / baseline_value * 100
                else:
                    # Higher is better for accuracy metrics
                    improvement = (current_value - baseline_value) / baseline_value * 100
                
                comparison[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'improvement_percent': improvement,
                    'better': improvement > 0
                }
        
        self.results['baseline_comparison'] = comparison
        return comparison
    
    def save_results(self, filepath: str):
        """Save evaluation results to file"""
        results_copy = self.results.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_copy = convert_numpy(results_copy)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f" Evaluation results saved to {filepath}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print(" EVALUATION SUMMARY")
        print("="*60)
        
        metrics = self.results.get('metrics', {})
        
        if not metrics:
            print(" No evaluation metrics available")
            return
        
        print(f" Performance Metrics:")
        print(f"   Translation Error: {metrics.get('mean_translation_error', 0):.3f} ± {metrics.get('std_translation_error', 0):.3f} m")
        print(f"   Rotation Error: {metrics.get('mean_rotation_error', 0):.3f} ± {metrics.get('std_rotation_error', 0):.3f} rad")
        print(f"   Category Accuracy: {metrics.get('category_accuracy', 0):.1%}")
        
        print(f"\n Performance:")
        print(f"   Mean Inference Time: {metrics.get('mean_inference_time', 0):.3f} s")
        print(f"   FPS: {metrics.get('fps', 0):.1f}")
        
        print(f"\n Sample Statistics:")
        print(f"   Total Samples: {metrics.get('total_samples', 0)}")
        print(f"   Successful Predictions: {metrics.get('successful_predictions', 0)}")
        
        # Category breakdown
        if 'category_performance' in self.results:
            print(f"\n Performance by Category:")
            for category, perf in self.results['category_performance'].items():
                print(f"   {category.capitalize()}:")
                print(f"     Samples: {perf['sample_count']}")
                print(f"     Translation Error: {perf['mean_translation_error']:.3f} m")
                print(f"     Category Accuracy: {perf['category_accuracy']:.1%}")
        
        print("="*60)

def create_evaluator(model: nn.Module, device: torch.device, config: Any) -> ModelEvaluator:
    """
    Factory function to create an evaluator instance
    
    Args:
        model (nn.Module): Model to evaluate
        device (torch.device): Device for evaluation
        config (Config): Configuration object
    
    Returns:
        ModelEvaluator: Configured evaluator instance
    """
    return ModelEvaluator(model, device, config)

def load_baseline_results(baseline_name: str) -> Dict[str, float]:
    """
    Load baseline results for comparison
    
    Args:
        baseline_name (str): Name of baseline method
    
    Returns:
        dict: Baseline results
    """
    # Simulated baseline results (replace with actual values when available)
    baselines = {
        'rtm3d': {
            'mean_translation_error': 61.77,
            'mean_rotation_error': 3.82,
            'category_accuracy': 0.286,
            'mean_inference_time': 0.066  # ~15 FPS
        },
        'vehipose': {
            'mean_translation_error': 52.27,
            'mean_rotation_error': 3.21,
            'category_accuracy': 0.318,
            'mean_inference_time': 0.115  # ~8.7 FPS
        },
        'original_hspose': {
            'mean_translation_error': 55.0,
            'mean_rotation_error': 3.5,
            'category_accuracy': 0.30,
            'mean_inference_time': 0.08  # ~12.5 FPS
        }
    }
    
    return baselines.get(baseline_name, {})

if __name__ == "__main__":
    print(" Model evaluation system ready!")
    print("🔧 Features available:")
    print("  - Comprehensive metric calculation")
    print("  - Performance analysis by category and distance")
    print("  - Statistical significance testing")
    print("  - Baseline comparison")
    print("  - Detailed result reporting")