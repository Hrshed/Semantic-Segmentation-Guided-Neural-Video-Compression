# compression_training/callbacks/csv_logging.py
import os
import csv
from typing import Dict, List
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class CSVLoggingCallback(pl.Callback):
    """Callback for logging training and validation metrics to CSV files."""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.csv_log_dir = None
        self.train_csv_path = None
        self.val_csv_path = None
        self.train_headers_written = False
        self.val_headers_written = False
        
        # Define headers for CSV files
        self.train_headers = [
            "epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", 
            "psnr", "mse", "lr_main", "lr_aux", "qp_avg"
        ]
        self.val_headers = [
            "epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", 
            "psnr", "mse"
        ]

    def setup(self, trainer, pl_module, stage=None):
        """Setup CSV logging paths and create header files."""
        if stage == "fit":
            self._setup_csv_logging(trainer)

    @rank_zero_only
    def _setup_csv_logging(self, trainer):
        """Setup CSV logging directories and files."""
        if trainer.logger is not None and hasattr(trainer.logger, 'log_dir'):
            experiment_log_dir = trainer.logger.log_dir
            if experiment_log_dir:
                self.csv_log_dir = os.path.join(experiment_log_dir, "csv_metrics")
                self.train_csv_path = os.path.join(self.csv_log_dir, "train_metrics.csv")
                self.val_csv_path = os.path.join(self.csv_log_dir, "val_metrics.csv")
                
                try:
                    os.makedirs(self.csv_log_dir, exist_ok=True)
                    
                    # Setup train CSV
                    if not os.path.exists(self.train_csv_path):
                        with open(self.train_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(self.train_headers)
                        self.train_headers_written = True
                        print(f"Train CSV log file created at: {self.train_csv_path}")
                    
                    # Setup validation CSV
                    if not os.path.exists(self.val_csv_path):
                        with open(self.val_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(self.val_headers)
                        self.val_headers_written = True
                        print(f"Validation CSV log file created at: {self.val_csv_path}")
                        
                except Exception as e:
                    print(f"Error setting up CSV logging: {e}")
                    self.train_csv_path = None
                    self.val_csv_path = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics to CSV on specified intervals."""
        if batch_idx % self.log_interval == 0:
            # Get current metrics from the trainer
            optimizers = trainer.optimizers
            if isinstance(optimizers, list) and len(optimizers) >= 2:
                lr_main = optimizers[0].param_groups[0]['lr']
                lr_aux = optimizers[1].param_groups[0]['lr']
            else:
                lr_main = lr_aux = 0.0
            
            # Get logged metrics
            logged_metrics = trainer.logged_metrics
            
            metrics_dict = {
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "loss": logged_metrics.get("train/loss", 0.0),
                "bpp": logged_metrics.get("train/bpp", 0.0),
                "bpp_y": logged_metrics.get("train/bpp_y", 0.0),
                "bpp_z": logged_metrics.get("train/bpp_z", 0.0),
                "psnr": logged_metrics.get("train/psnr", 0.0),
                "mse": logged_metrics.get("train/mse", 0.0),
                "lr_main": lr_main,
                "lr_aux": lr_aux,
                "qp_avg": logged_metrics.get("train/qp", 0.0)
            }
            
            self._log_metrics_to_csv(
                self.train_csv_path, 
                self.train_headers, 
                metrics_dict, 
                "train_headers_written"
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics to CSV at epoch end."""
        metrics = trainer.callback_metrics
        
        log_data = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "loss": metrics.get("val/loss", 0.0),
            "bpp": metrics.get("val/bpp", 0.0),
            "bpp_y": metrics.get("val/bpp_y", 0.0),
            "bpp_z": metrics.get("val/bpp_z", 0.0),
            "psnr": metrics.get("val/psnr", 0.0),
            "mse": metrics.get("val/mse", 0.0),
        }
        
        self._log_metrics_to_csv(
            self.val_csv_path, 
            self.val_headers, 
            log_data, 
            "val_headers_written"
        )

    @rank_zero_only
    def _log_metrics_to_csv(self, file_path: str, headers: List[str], 
                           metrics_dict: Dict, headers_written_flag: str):
        """Write metrics to CSV file."""
        if not file_path:
            return
        
        try:
            file_exists = os.path.exists(file_path)
            headers_written = getattr(self, headers_written_flag)
            
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists or not headers_written:
                    writer.writerow(headers)
                    setattr(self, headers_written_flag, True)
                
                # Convert tensor values to floats
                row_values = []
                for header in headers:
                    value = metrics_dict.get(header, "")
                    if hasattr(value, 'item'):  # Handle tensors
                        value = value.item()
                    row_values.append(value)
                
                writer.writerow(row_values)
                
        except Exception as e:
            print(f"Error writing to CSV log {os.path.basename(file_path)}: {e}")