import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

# Set style for plots with fallback
try:
    # Try different style options
    available_styles = plt.style.available
    if 'seaborn' in available_styles:
        plt.style.use('seaborn')
    elif 'seaborn-v0_8' in available_styles:  # For newer matplotlib versions
        plt.style.use('seaborn-v0_8')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    else:
        plt.style.use('default')
    
    # Set seaborn theme with error handling
    try:
        sns.set_theme(style="whitegrid")
    except Exception as e:
        print(f"Warning: Could not set seaborn theme: {e}")
        sns.set()  # Use default seaborn settings
        
except Exception as e:
    print(f"Warning: Could not set plot style: {e}")
    # Continue with default style

class MetricLogger:
    def __init__(self, log_dir='runs', exp_name='exp', resume=False):
        """
        Initialize the logger
        
        Args:
            log_dir (str): Directory to save logs
            exp_name (str): Experiment name
            resume (bool): Whether to resume from existing log
        """
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.best_metrics = {}
        
        # Create log directory
        if not resume:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.log_dir / f"{exp_name}_{timestamp}"
        else:
            self.run_dir = self.log_dir / exp_name
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.weights_dir = self.run_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.run_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        
        # Save config
        self.config = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'log_dir': str(self.run_dir)
        }
        
    def update_metrics(self, metrics_dict, epoch=None, step=None, phase='train'):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            metric_name = f'{phase}/{key}'
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[metric_name].append((step, value))
            
            # Update epoch metrics
            if epoch is not None:
                if len(self.epoch_metrics[metric_name]) <= epoch:
                    self.epoch_metrics[metric_name].append([])
                self.epoch_metrics[metric_name][epoch].append(value)
            
            # Update TensorBoard
            if step is not None:
                self.writer.add_scalar(metric_name, value, step)
    
    def log_epoch(self, epoch, model, optimizer, metrics_dict, phase='val'):
        """Log metrics at the end of an epoch"""
        log_str = f"Epoch: {epoch} | "
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                log_str += f"{key}: {value:.4f} | "
            else:
                log_str += f"{key}: {value} | "
        
        # Print to console
        print(log_str)
        
        # Save to log file
        with open(self.run_dir / 'training_log.txt', 'a') as f:
            f.write(log_str + '\n')
        
        # Save best model
        if phase == 'val' and 'mAP' in metrics_dict:
            current_map = metrics_dict['mAP']
            if 'best_mAP' not in self.best_metrics or current_map > self.best_metrics['best_mAP']:
                self.best_metrics['best_mAP'] = current_map
                self.best_metrics['best_epoch'] = epoch
                self.save_checkpoint(model, optimizer, epoch, is_best=True)
        
        # Save checkpoint
        if phase == 'val' and epoch % 5 == 0:  # Save every 5 epochs
            self.save_checkpoint(model, optimizer, epoch)
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': self.metrics,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        if is_best:
            torch.save(state, self.weights_dir / 'best_model.pth')
        torch.save(state, self.weights_dir / f'model_epoch_{epoch}.pth')
    
    def log_metrics(self, metrics, step, phase='train'):
        """Log metrics to TensorBoard, console, and CSV"""
        # Ensure plots directory exists
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Log to TensorBoard
        for k, v in metrics.items():
            if v is not None:
                self.writer.add_scalar(f'{phase}/{k}', v, step)
        
        # Update metrics history
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append((step, v))
        
        # Save metrics to CSV
        self._save_metrics_to_csv(metrics, step, phase)
        
        # Print to console
        if step % 10 == 0:  # Print every 10 steps
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            print(f'Step {step}: {metrics_str}')
        
        # Generate and save plots every 100 steps
        if step % 100 == 0:
            self._generate_plots(phase)
    
    def _save_metrics_to_csv(self, metrics, step, phase):
        """Save metrics to CSV file"""
        csv_path = os.path.join(self.log_dir, 'metrics.csv')
        
        # Prepare data for CSV
        data = {'step': step, 'phase': phase}
        data.update(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Append to existing CSV or create new
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def _generate_plots(self, phase):
        """Generate and save plots of metrics"""
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a figure for loss
        plt.figure(figsize=(12, 6))
        
        # Plot training and validation losses if available
        for metric in ['loss', 'mAP']:
            if f'train_{metric}' in self.metrics and f'val_{metric}' in self.metrics:
                plt.figure(figsize=(12, 6))
                
                # Plot training metric
                steps, values = zip(*self.metrics[f'train_{metric}'])
                plt.plot(steps, values, label=f'Train {metric}')
                
                # Plot validation metric
                val_steps, val_values = zip(*self.metrics[f'val_{metric}'])
                plt.plot(val_steps, val_values, label=f'Validation {metric}')
                
                plt.title(f'{metric.capitalize()} over Training')
                plt.xlabel('Step')
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plot_path = os.path.join(plots_dir, f'{metric}_plot.png')
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
    
    def log_config(self, config_dict):
        """Log configuration parameters"""
        self.config.update(config_dict)
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def close(self):
        """Close the logger and save final metrics"""
        # Generate final plots
        self._generate_plots('final')
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nTraining complete. Logs and plots saved to: {self.log_dir}")
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save best metrics
        with open(self.run_dir / 'best_metrics.json', 'w') as f:
            json.dump(self.best_metrics, f, indent=4)
        
        # Generate final plots
        try:
            self.plot_metrics()
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Print summary
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best mAP: {self.best_metrics.get('best_mAP', 0):.4f} at epoch {self.best_metrics.get('best_epoch', 0)}")
        print(f"Logs and models saved to: {self.run_dir}")
    
    def plot_metrics(self):
        """Generate training plots"""
        try:
            if not self.metrics_history:
                print("No metrics to plot")
                return
            
            # Create plots directory
            plots_dir = self.run_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot loss curves
            epochs = list(range(len(self.metrics_history)))
            train_losses = [m.get('loss', 0) for m in self.metrics_history]
            val_losses = [m.get('val_loss', 0) for m in self.metrics_history]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss')
            plt.plot(epochs, val_losses, 'r-', label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot mAP
            plt.subplot(2, 2, 2)
            maps = [m.get('mAP', 0) for m in self.metrics_history]
            plt.plot(epochs, maps, 'g-', label='mAP')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision')
            plt.legend()
            plt.grid(True)
            
            # Plot learning rate
            plt.subplot(2, 2, 3)
            lrs = [m.get('lr', 0) for m in self.metrics_history]
            plt.plot(epochs, lrs, 'm-', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            
            # Plot AP for each class
            plt.subplot(2, 2, 4)
            ap_fire = [m.get('AP_fire', 0) for m in self.metrics_history]
            ap_smoke = [m.get('AP_smoke', 0) for m in self.metrics_history]
            plt.plot(epochs, ap_fire, 'orange', label='AP Fire')
            plt.plot(epochs, ap_smoke, 'purple', label='AP Smoke')
            plt.xlabel('Epoch')
            plt.ylabel('AP')
            plt.title('Average Precision by Class')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Training plots saved to: {plots_dir / 'training_metrics.png'}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")

# Example usage:
if __name__ == "__main__":
    print("This is a logger module. Import and use it in your training script.")
    print("Example usage in training script:")
    print("""
    # Initialize logger
    logger = MetricLogger(log_dir='runs', exp_name='fire_smoke_detection')
    
    # Log configuration
    config = {
        'model': 'EnhancedFasterRCNN',
        'backbone': 'ResNeXt101-32x8d-FPN',
        'batch_size': 4,
        'learning_rate': 0.0025,
        'num_epochs': 24,
        'optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'image_size': 640
    }
    logger.log_config(config)
    
    # In your training loop:
    # for epoch in range(num_epochs):
    #     # Training steps...
    #     metrics = {'loss': train_loss, 'lr': current_lr}
    #     logger.update_metrics(metrics, epoch=epoch, step=current_step, phase='train')
    #     
    #     # Validation
    #     val_metrics = {
    #         'loss': val_loss,
    #         'mAP': val_map,
    #         'AP_fire': val_ap_fire,
    #         'AP_smoke': val_ap_smoke
    #     }
    #     logger.update_metrics(val_metrics, epoch=epoch, step=current_step, phase='val')
    #     
    #     # Log end of epoch (only save checkpoint if model and optimizer are provided)
    #     logger.log_epoch(epoch, model, optimizer, val_metrics, phase='val')
    # 
    # # Close logger when done
    # logger.close()
    """)