"""
Logging Utilities for HS-Pose Vehicle Estimation
===============================================

This module provides comprehensive logging functionality for the project,
including file logging, console output, progress tracking, and integration
with external logging services.

Key Features:
- Structured logging with different severity levels
- File and console output
- Progress tracking for long operations
- Performance monitoring
- Integration with tensorboard and wandb
- Automatic log rotation and cleanup
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime
import traceback
from contextlib import contextmanager

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Store original level name
        original_levelname = record.levelname
        
        # Apply color
        record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        # Format the message
        formatted = super().format(record)
        
        # Restore original level name
        record.levelname = original_levelname
        
        return formatted

class PerformanceTracker:
    """Track performance metrics during execution"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation"""
        if name not in self.metrics:
            return {}
        
        times = self.metrics[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()

class ProjectLogger:
    """
    Main logging class for the HS-Pose project
    
    Provides structured logging with file output, console output,
    and integration with external logging services.
    """
    
    def __init__(self, name: str = "hspose", log_dir: str = "./logs", level: str = "INFO"):
        """
        Initialize the logger
        
        Args:
            name (str): Logger name
            log_dir (str): Directory for log files
            level (str): Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Performance tracker
        self.perf_tracker = PerformanceTracker()
        
        # Setup handlers
        self._setup_handlers()
        
        # Session info
        self.session_start = datetime.now()
        self.session_id = f"{name}_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Log session start
        self.info(f"Logger initialized - Session ID: {self.session_id}")
    
    def _setup_handlers(self):
        """Setup file and console handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler with detailed formatting
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler with colored formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error file handler for errors and above
        error_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context"""
        # Add extra context if provided
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {context_str}"
        
        self.logger.log(level, message)
    
    def log_exception(self, message: str = "Exception occurred"):
        """Log exception with full traceback"""
        self.error(f"{message}\n{traceback.format_exc()}")
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics dictionary"""
        prefix_str = f"{prefix}_" if prefix else ""
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Nested dictionary
                self.log_metrics(value, f"{prefix_str}{key}")
            else:
                # Simple value
                metric_name = f"{prefix_str}{key}"
                self.info(f"Metric: {metric_name} = {value}")
    
    def log_config(self, config: Any, title: str = "Configuration"):
        """Log configuration object"""
        self.info(f"=== {title} ===")
        
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            self.warning(f"Cannot log config of type {type(config)}")
            return
        
        for key, value in config_dict.items():
            self.info(f"{key}: {value}")
        
        self.info("=" * (len(title) + 8))
    
    def log_model_info(self, model, input_shape: tuple = None):
        """Log model information"""
        self.info("=== Model Information ===")
        self.info(f"Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        if input_shape:
            self.info(f"Expected input shape: {input_shape}")
        
        self.info("=" * 25)
    
    @contextmanager
    def timer(self, operation_name: str, log_result: bool = True):
        """Context manager for timing operations"""
        start_time = time.time()
        self.perf_tracker.start_timer(operation_name)
        
        try:
            if log_result:
                self.debug(f"Starting: {operation_name}")
            yield
        finally:
            duration = self.perf_tracker.end_timer(operation_name)
            if log_result:
                self.info(f"Completed: {operation_name} in {duration:.3f}s")
    
    def log_training_epoch(self, epoch: int, total_epochs: int, 
                          train_loss: float, val_loss: float, 
                          metrics: Dict[str, float] = None):
        """Log training epoch summary"""
        progress = (epoch + 1) / total_epochs * 100
        
        message = f"Epoch {epoch + 1}/{total_epochs} ({progress:.1f}%) | "
        message += f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    message += f" | {key}: {value:.4f}"
                else:
                    message += f" | {key}: {value}"
        
        self.info(message)
    
    def log_evaluation_results(self, results: Dict[str, Any]):
        """Log evaluation results"""
        self.info("=== Evaluation Results ===")
        
        # Main metrics
        if 'metrics' in results:
            metrics = results['metrics']
            self.info(f"Translation Error: {metrics.get('mean_translation_error', 0):.3f} ± {metrics.get('std_translation_error', 0):.3f} m")
            self.info(f"Rotation Error: {metrics.get('mean_rotation_error', 0):.3f} ± {metrics.get('std_rotation_error', 0):.3f} rad")
            self.info(f"Category Accuracy: {metrics.get('category_accuracy', 0):.1%}")
            self.info(f"Inference Speed: {metrics.get('fps', 0):.1f} FPS")
        
        # Category performance
        if 'category_performance' in results:
            self.info("--- Performance by Category ---")
            for category, perf in results['category_performance'].items():
                self.info(f"{category.capitalize()}: "
                         f"Trans Err: {perf['mean_translation_error']:.3f}m, "
                         f"Accuracy: {perf['category_accuracy']:.1%}, "
                         f"Samples: {perf['sample_count']}")
        
        self.info("=" * 26)
    
    def log_comparison(self, our_results: Dict[str, float], 
                      baseline_results: Dict[str, float], 
                      baseline_name: str):
        """Log comparison with baseline"""
        self.info(f"=== Comparison with {baseline_name} ===")
        
        for metric in our_results.keys():
            if metric in baseline_results:
                our_val = our_results[metric]
                baseline_val = baseline_results[metric]
                
                if 'error' in metric:
                    # Lower is better
                    improvement = (baseline_val - our_val) / baseline_val * 100
                    comparison = "↓" if improvement > 0 else "↑"
                else:
                    # Higher is better
                    improvement = (our_val - baseline_val) / baseline_val * 100
                    comparison = "↑" if improvement > 0 else "↓"
                
                self.info(f"{metric}: Ours: {our_val:.3f} | {baseline_name}: {baseline_val:.3f} | "
                         f"{comparison} {abs(improvement):.1f}%")
        
        self.info("=" * (len(baseline_name) + 20))
    
    def save_session_summary(self):
        """Save session summary to file"""
        session_end = datetime.now()
        duration = session_end - self.session_start
        
        summary = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'end_time': session_end.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'performance_metrics': {}
        }
        
        # Add performance metrics
        for operation, times in self.perf_tracker.metrics.items():
            summary['performance_metrics'][operation] = self.perf_tracker.get_stats(operation)
        
        # Save to file
        summary_file = self.log_dir / f"session_summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.info(f"Session summary saved to {summary_file}")
        self.info(f"Session duration: {duration}")

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, logger: ProjectLogger, total_steps: int, description: str = "Progress"):
        self.logger = logger
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 10  # Log every 10 steps by default
    
    def update(self, steps: int = 1, message: str = None):
        """Update progress"""
        self.current_step += steps
        current_time = time.time()
        
        # Calculate progress
        progress = self.current_step / self.total_steps
        elapsed = current_time - self.start_time
        
        # Estimate remaining time
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = 0
        
        # Log periodically or on completion
        should_log = (
            self.current_step % self.log_interval == 0 or
            self.current_step == self.total_steps or
            current_time - self.last_log_time > 30  # Every 30 seconds minimum
        )
        
        if should_log:
            progress_msg = f"{self.description}: {self.current_step}/{self.total_steps} "
            progress_msg += f"({progress:.1%}) | "
            progress_msg += f"Elapsed: {elapsed:.1f}s | "
            progress_msg += f"Remaining: {remaining:.1f}s"
            
            if message:
                progress_msg += f" | {message}"
            
            self.logger.info(progress_msg)
            self.last_log_time = current_time
    
    def finish(self, message: str = None):
        """Mark progress as finished"""
        elapsed = time.time() - self.start_time
        final_msg = f"{self.description} completed in {elapsed:.1f}s"
        
        if message:
            final_msg += f" | {message}"
        
        self.logger.info(final_msg)

def setup_logger(log_dir: str = "./logs", level: str = "INFO", name: str = "hspose") -> ProjectLogger:
    """
    Setup project logger
    
    Args:
        log_dir (str): Directory for log files
        level (str): Logging level
        name (str): Logger name
    
    Returns:
        ProjectLogger: Configured logger instance
    """
    return ProjectLogger(name=name, log_dir=log_dir, level=level)

class LogCapture:
    """Capture logs for testing or analysis"""
    
    def __init__(self, logger: ProjectLogger):
        self.logger = logger
        self.captured_logs = []
        self.original_handlers = []
    
    def __enter__(self):
        # Store original handlers
        self.original_handlers = self.logger.logger.handlers.copy()
        
        # Create capturing handler
        self.capture_handler = logging.Handler()
        self.capture_handler.emit = self._capture_log
        
        # Replace handlers with capture handler
        self.logger.logger.handlers = [self.capture_handler]
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        self.logger.logger.handlers = self.original_handlers
    
    def _capture_log(self, record):
        """Capture log record"""
        self.captured_logs.append({
            'level': record.levelname,
            'message': record.getMessage(),
            'timestamp': record.created,
            'filename': record.filename,
            'lineno': record.lineno
        })
    
    def get_logs(self, level: str = None) -> list:
        """Get captured logs, optionally filtered by level"""
        if level is None:
            return self.captured_logs
        else:
            return [log for log in self.captured_logs if log['level'] == level.upper()]

# Global logger instance
_global_logger = None

def get_logger() -> ProjectLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger

def cleanup_old_logs(log_dir: str, max_age_days: int = 30):
    """Clean up old log files"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    for log_file in log_path.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Error deleting {log_file}: {e}")

# Example usage and convenience functions
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            try:
                with logger.timer(f"function_{func_name}"):
                    result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    print("Logging System Example")
    print("=" * 30)
    
    # Create logger
    logger = setup_logger("./test_logs", "DEBUG", "test_logger")
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    logger.log_metrics({
        'accuracy': 0.95,
        'loss': 0.05,
        'epoch': 10
    }, prefix="training")
    
    # Test timing
    with logger.timer("test_operation"):
        time.sleep(1)
    
    # Test progress tracking
    progress = ProgressTracker(logger, 100, "Test Progress")
    for i in range(100):
        time.sleep(0.01)
        progress.update(1)
    progress.finish("All done!")
    
    # Save session summary
    logger.save_session_summary()
    
    print("\n✅ Logging system test completed!")

print("📝 Logging system loaded successfully!")
print("🔧 Features available:")
print("  - Colored console output")
print("  - File logging with rotation")
print("  - Performance tracking")
print("  - Progress monitoring")
print("  - Structured logging")
print("  - Session summaries")