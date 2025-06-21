"""
Training Pipeline for Vehicle Pose Estimation
============================================

This module provides a comprehensive training pipeline for the vehicle pose estimation
model, including training loop, validation, checkpointing, and progress tracking.

Key Features:
- Multi-task loss optimization (pose + category + confidence + dimensions)
- Learning rate scheduling and early stopping
- Comprehensive metrics tracking
- Model checkpointing and resuming
- Mixed precision training support
- Tensorboard and wandb integration
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

class VehiclePoseTrainer:
    """
    Comprehensive training pipeline for vehicle pose estimation
    
    This class handles the complete training process including loss computation,
    optimization, validation, checkpointing, and progress tracking.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Any):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The model to train
            device (torch.device): Device to run training on
            config (Config): Configuration object
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Move model to device
        self.model.to(device)
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Training state tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.metrics_history = {
            'train_pose_loss': [],
            'train_category_loss': [],
            'train_confidence_loss': [],
            'train_dimension_loss': [],
            'val_pose_loss': [],
            'val_category_loss': [],
            'val_confidence_loss': [],
            'val_dimension_loss': [],
            'val_translation_error': [],
            'val_rotation_error': [],
            'val_category_accuracy': []
        }
        
        # Setup logging
        self.setup_logging()
        
        # Early stopping
        self.early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = getattr(config, 'EARLY_STOPPING_MIN_DELTA', 0.001)
        
    def _setup_loss_functions(self):
        """Initialize loss functions with appropriate weights"""
        self.pose_loss_fn = nn.MSELoss(reduction='mean')
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.BCELoss()
        self.dimension_loss_fn = nn.MSELoss(reduction='mean')
        
        # Loss weights from config
        self.loss_weights = {
            'pose': getattr(self.config, 'POSE_LOSS_WEIGHT', 1.0),
            'category': getattr(self.config, 'CATEGORY_LOSS_WEIGHT', 0.5),
            'confidence': getattr(self.config, 'CONFIDENCE_LOSS_WEIGHT', 0.3),
            'dimension': getattr(self.config, 'DIMENSION_LOSS_WEIGHT', 0.2)
        }
        
        # Distance-based weighting for pose loss
        self.use_distance_weighting = getattr(self.config, 'USE_DISTANCE_WEIGHTING', True)
        
    def setup_logging(self):
        """Setup logging for training progress"""
        # Tensorboard logging
        if getattr(self.config, 'USE_TENSORBOARD', True):
            log_dir = Path(self.config.LOG_DIR) / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Weights & Biases logging
        self.use_wandb = getattr(self.config, 'USE_WANDB', False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=getattr(self.config, 'WANDB_PROJECT', 'hspose-vehicle'),
                    name=f"run_{int(time.time())}",
                    config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss for vehicle pose estimation
        
        Args:
            outputs (dict): Model predictions
            targets (dict): Ground truth targets
        
        Returns:
            dict: Individual and total losses
        """
        losses = {}
        
        # Extract predictions and targets
        pred_pose = outputs['pose']
        pred_category = outputs['category']
        pred_confidence = outputs['confidence']
        pred_dimensions = outputs.get('dimensions', None)
        
        gt_pose = targets['poses']
        gt_category = targets['categories']
        gt_confidence = targets['confidences']
        gt_dimensions = targets.get('dimensions', None)
        
        # Ensure correct shapes and devices
        pred_pose = pred_pose.to(self.device)
        pred_category = pred_category.to(self.device)
        pred_confidence = pred_confidence.to(self.device)
        
        gt_pose = gt_pose.to(self.device)
        gt_category = gt_category.to(self.device)
        gt_confidence = gt_confidence.to(self.device)
        
        # Reshape predictions if necessary
        if len(pred_pose.shape) > 2:
            pred_pose = pred_pose.view(pred_pose.shape[0], -1)
        if len(pred_category.shape) > 2:
            pred_category = pred_category.view(pred_category.shape[0], -1)
        if len(pred_confidence.shape) > 1:
            pred_confidence = pred_confidence.view(-1)
        
        # Ensure pose dimensions match
        if pred_pose.shape[1] != gt_pose.shape[1]:
            if pred_pose.shape[1] > gt_pose.shape[1]:
                pred_pose = pred_pose[:, :gt_pose.shape[1]]
            else:
                padding = torch.zeros(pred_pose.shape[0], gt_pose.shape[1] - pred_pose.shape[1]).to(self.device)
                pred_pose = torch.cat([pred_pose, padding], dim=1)
        
        # Pose loss with optional distance weighting
        if self.use_distance_weighting:
            # Weight closer vehicles more heavily
            distances = torch.norm(gt_pose[:, :3], dim=1)
            distance_weights = 1.0 / (1.0 + distances / 50.0)  # Normalize by 50m
            
            pose_errors = ((pred_pose - gt_pose) ** 2).mean(dim=1)
            pose_loss = (pose_errors * distance_weights).mean()
        else:
            pose_loss = self.pose_loss_fn(pred_pose, gt_pose)
        
        losses['pose_loss'] = pose_loss
        
        # Category classification loss
        if gt_category.dtype != torch.long:
            gt_category = gt_category.long()
        losses['category_loss'] = self.category_loss_fn(pred_category, gt_category)
        
        # Confidence loss
        if gt_confidence.dtype != torch.float:
            gt_confidence = gt_confidence.float()
        if pred_confidence.dtype != torch.float:
            pred_confidence = pred_confidence.float()
        losses['confidence_loss'] = self.confidence_loss_fn(pred_confidence, gt_confidence)
        
        # Dimension loss (if available)
        if pred_dimensions is not None and gt_dimensions is not None:
            pred_dimensions = pred_dimensions.to(self.device)
            gt_dimensions = gt_dimensions.to(self.device)
            
            if len(pred_dimensions.shape) > 2:
                pred_dimensions = pred_dimensions.view(pred_dimensions.shape[0], -1)
            
            losses['dimension_loss'] = self.dimension_loss_fn(pred_dimensions, gt_dimensions)
        else:
            losses['dimension_loss'] = torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['pose'] * losses['pose_loss'] +
            self.loss_weights['category'] * losses['category_loss'] +
            self.loss_weights['confidence'] * losses['confidence_loss'] +
            self.loss_weights['dimension'] * losses['dimension_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                  optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Perform single training step
        
        Args:
            batch (dict): Training batch
            optimizer (torch.optim.Optimizer): Optimizer
        
        Returns:
            dict: Loss values
        """
        self.model.train()
        
        # Move batch to device
        images = batch['images'].to(self.device)
        targets = {
            'poses': batch['poses'].to(self.device),
            'categories': batch['categories'].to(self.device),
            'confidences': batch['confidences'].to(self.device)
        }
        
        # Add dimensions if available
        if 'dimensions' in batch:
            targets['dimensions'] = batch['dimensions'].to(self.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images)
        
        # Compute loss
        losses = self.compute_loss(outputs, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        if hasattr(self.config, 'GRADIENT_CLIP_NORM'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.GRADIENT_CLIP_NORM
            )
        
        # Optimizer step
        optimizer.step()
        
        # Update global step
        self.global_step += 1
        
        # Convert losses to float for logging
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_losses = []
        translation_errors = []
        rotation_errors = []
        category_accuracies = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                targets = {
                    'poses': batch['poses'].to(self.device),
                    'categories': batch['categories'].to(self.device),
                    'confidences': batch['confidences'].to(self.device)
                }
                
                if 'dimensions' in batch:
                    targets['dimensions'] = batch['dimensions'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                losses = self.compute_loss(outputs, targets)
                val_losses.append({k: v.item() for k, v in losses.items()})
                
                # Compute additional metrics
                pred_poses = outputs['pose']
                gt_poses = targets['poses']
                
                if len(pred_poses.shape) > 2:
                    pred_poses = pred_poses.view(pred_poses.shape[0], -1)
                
                # Ensure dimensions match
                if pred_poses.shape[1] != gt_poses.shape[1]:
                    if pred_poses.shape[1] > gt_poses.shape[1]:
                        pred_poses = pred_poses[:, :gt_poses.shape[1]]
                
                # Translation error (L2 distance)
                trans_error = torch.norm(pred_poses[:, :3] - gt_poses[:, :3], dim=1)
                translation_errors.extend(trans_error.cpu().numpy())
                
                # Rotation error (angular difference)
                rot_error = torch.norm(pred_poses[:, 3:6] - gt_poses[:, 3:6], dim=1)
                rotation_errors.extend(rot_error.cpu().numpy())
                
                # Category accuracy
                pred_categories = torch.argmax(outputs['category'], dim=1)
                gt_categories = targets['categories']
                cat_accuracy = (pred_categories == gt_categories).float()
                category_accuracies.extend(cat_accuracy.cpu().numpy())
        
        # Aggregate validation metrics
        val_metrics = {}
        
        # Average losses
        for loss_key in val_losses[0].keys():
            val_metrics[f'val_{loss_key}'] = np.mean([batch[loss_key] for batch in val_losses])
        
        # Additional metrics
        val_metrics['val_translation_error'] = np.mean(translation_errors)
        val_metrics['val_rotation_error'] = np.mean(rotation_errors)
        val_metrics['val_category_accuracy'] = np.mean(category_accuracies)
        
        return val_metrics
    
    def train_epoch(self, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (optional): Learning rate scheduler
        
        Returns:
            dict: Average training metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch, optimizer)
            epoch_losses.append(losses)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss']:.4f}",
                'Pose': f"{losses['pose_loss']:.4f}",
                'Cat': f"{losses['category_loss']:.4f}",
                'Conf': f"{losses['confidence_loss']:.4f}"
            })
            
            # Log batch metrics
            if self.writer and batch_idx % self.config.LOG_INTERVAL == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'train_batch/{loss_name}', loss_value, self.global_step)
                self.writer.add_scalar('train_batch/learning_rate', 
                                     optimizer.param_groups[0]['lr'], self.global_step)
            
            if self.wandb and batch_idx % self.config.LOG_INTERVAL == 0:
                wandb_logs = {f'train_batch/{k}': v for k, v in losses.items()}
                wandb_logs['train_batch/learning_rate'] = optimizer.param_groups[0]['lr']
                self.wandb.log(wandb_logs, step=self.global_step)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs the metric value
                pass  # Will be updated after validation
            else:
                scheduler.step()
        
        # Calculate average losses for epoch
        avg_losses = {}
        for loss_key in epoch_losses[0].keys():
            avg_losses[f'train_{loss_key}'] = np.mean([batch[loss_key] for batch in epoch_losses])
        
        return avg_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        
        Returns:
            dict: Training results and metrics
        """
        print(f"🚀 Starting training for {self.config.NUM_EPOCHS} epochs")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        # Setup optimizer
        optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        scheduler = self._setup_scheduler(optimizer, len(train_loader))
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate scheduler if needed
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_total_loss'])
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['learning_rate'] = current_lr
            epoch_metrics['epoch_time'] = time.time() - epoch_start_time
            
            # Update metrics history
            for key, value in epoch_metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:")
            print(f"  Train Loss: {train_metrics['train_total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_total_loss']:.4f}")
            print(f"  Val Trans Error: {val_metrics['val_translation_error']:.3f}m")
            print(f"  Val Category Acc: {val_metrics['val_category_accuracy']:.1%}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_metrics['epoch_time']:.1f}s")
            
            # Save checkpoint
            is_best = val_metrics['val_total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_total_loss']
                self.best_metrics = val_metrics.copy()
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save model checkpoint
            if self.config.SAVE_CHECKPOINTS:
                self._save_checkpoint(epoch_metrics, is_best)
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                print(f"   No improvement for {self.early_stopping_patience} epochs")
                break
        
        # Training complete
        total_training_time = time.time() - training_start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️ Total training time: {total_training_time / 3600:.2f} hours")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Generate training summary plots
        self._generate_training_plots()
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        if self.wandb:
            self.wandb.finish()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'total_epochs': self.current_epoch + 1,
            'total_training_time': total_training_time,
            'metrics_history': self.metrics_history
        }
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        # Separate parameters for different components
        pose_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'pose' in name.lower():
                pose_params.append(param)
            else:
                other_params.append(param)
        
        # Use different learning rates
        base_lr = self.config.LEARNING_RATE
        optimizer = torch.optim.Adam([
            {'params': pose_params, 'lr': base_lr, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': other_params, 'lr': base_lr * 0.1, 'weight_decay': self.config.WEIGHT_DECAY}
        ])
        
        return optimizer
    
    def _setup_scheduler(self, optimizer: torch.optim.Optimizer, 
                        steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        scheduler_type = getattr(self.config, 'SCHEDULER_TYPE', 'step')
        
        if scheduler_type == 'step':
            step_size = getattr(self.config, 'SCHEDULER_STEP_SIZE', 20)
            gamma = getattr(self.config, 'SCHEDULER_GAMMA', 0.5)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'plateau':
            patience = getattr(self.config, 'SCHEDULER_PATIENCE', 5)
            factor = getattr(self.config, 'SCHEDULER_FACTOR', 0.5)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor, verbose=True
            )
        
        elif scheduler_type == 'cosine':
            T_max = self.config.NUM_EPOCHS
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        else:
            # No scheduler
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)
    
    def _log_epoch_metrics(self, metrics: Dict[str, float]):
        """Log epoch metrics to tensorboard and wandb"""
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch/{key}', value, self.current_epoch)
        
        if self.wandb:
            wandb_logs = {f'epoch/{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb.log(wandb_logs, step=self.current_epoch)
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'metrics': metrics
        }
        
        # Save regular checkpoint
        if (self.current_epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = Path(self.config.MODEL_SAVE_PATH) / f'checkpoint_epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.MODEL_SAVE_PATH) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save latest checkpoint
        latest_path = Path(self.config.MODEL_SAVE_PATH) / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def _generate_training_plots(self):
        """Generate training progress plots"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Training Progress', fontsize=16)
            
            epochs = range(1, len(self.metrics_history['train_total_loss']) + 1)
            
            # Loss curves
            axes[0, 0].plot(epochs, self.metrics_history['train_total_loss'], label='Train')
            axes[0, 0].plot(epochs, self.metrics_history['val_total_loss'], label='Validation')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Pose loss
            axes[0, 1].plot(epochs, self.metrics_history['train_pose_loss'], label='Train')
            axes[0, 1].plot(epochs, self.metrics_history['val_pose_loss'], label='Validation')
            axes[0, 1].set_title('Pose Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Category accuracy
            axes[0, 2].plot(epochs, self.metrics_history['val_category_accuracy'])
            axes[0, 2].set_title('Category Accuracy')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].grid(True)
            
            # Translation error
            axes[1, 0].plot(epochs, self.metrics_history['val_translation_error'])
            axes[1, 0].set_title('Translation Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Error (m)')
            axes[1, 0].grid(True)
            
            # Rotation error
            axes[1, 1].plot(epochs, self.metrics_history['val_rotation_error'])
            axes[1, 1].set_title('Rotation Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Error (rad)')
            axes[1, 1].grid(True)
            
            # Learning rate
            axes[1, 2].plot(epochs, self.learning_rates)
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.config.PLOTS_SAVE_PATH) / 'training_progress.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate training plots: {e}")
    
    def save_results(self, filepath: str):
        """Save training results to file"""
        results = {
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'metrics_history': self.metrics_history,
            'total_epochs': self.current_epoch + 1,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
        
        print(f"💾 Training results saved to {filepath}")

def create_trainer(model: nn.Module, device: torch.device, config: Any) -> VehiclePoseTrainer:
    """
    Factory function to create a trainer instance
    
    Args:
        model (nn.Module): Model to train
        device (torch.device): Device for training
        config (Config): Configuration object
    
    Returns:
        VehiclePoseTrainer: Configured trainer instance
    """
    return VehiclePoseTrainer(model, device, config)

def resume_training(checkpoint_path: str, model: nn.Module, 
                   device: torch.device, config: Any) -> VehiclePoseTrainer:
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to resume training
        device (torch.device): Device for training
        config (Config): Configuration object
    
    Returns:
        VehiclePoseTrainer: Trainer instance with loaded state
    """
    trainer = VehiclePoseTrainer(model, device, config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        trainer.current_epoch = checkpoint.get('epoch', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.metrics_history = checkpoint.get('metrics_history', trainer.metrics_history)
        
        print(f"✅ Resumed training from epoch {trainer.current_epoch + 1}")
        print(f"🏆 Best validation loss: {trainer.best_val_loss:.4f}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Starting training from scratch")
    
    return trainer

if __name__ == "__main__":
    print("🏋️ Vehicle pose training pipeline ready!")
    print("🔧 Features available:")
    print("  - Multi-task loss optimization")
    print("  - Learning rate scheduling")
    print("  - Early stopping")
    print("  - Comprehensive metrics tracking")
    print("  - Tensorboard and wandb integration")
    print("  - Model checkpointing and resuming")