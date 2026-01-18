# compression_training/callbacks/image_logging.py
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from compression_training.utils.compression import ycbcr2rgb, prepare_image_for_logging


class ImageLoggingCallback(pl.Callback):
    """Callback for logging original and reconstructed images to TensorBoard."""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training images at specified intervals."""
        if batch_idx % self.log_interval == 0 and batch_idx > 0:
            self._log_training_images(trainer, pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Log validation images for first batch only."""
        if batch_idx == 0:
            self._log_validation_images(trainer, pl_module, batch)

    @rank_zero_only
    def _log_training_images(self, trainer, pl_module, batch):
        """Log training images to TensorBoard."""
        try:
            # Extract first frame from first sequence in batch
            original_frame, reconstructed_frame = self._get_frame_pair(pl_module, batch, "train")
            if original_frame is not None and reconstructed_frame is not None:
                self._log_images(trainer, original_frame, reconstructed_frame, "train")
        except Exception as e:
            print(f"Error during training image logging: {e}")

    @rank_zero_only
    def _log_validation_images(self, trainer, pl_module, batch):
        """Log validation images to TensorBoard."""
        try:
            # Extract first frame from first sequence in batch
            original_frame, reconstructed_frame = self._get_frame_pair(pl_module, batch, "val")
            if original_frame is not None and reconstructed_frame is not None:
                self._log_images(trainer, original_frame, reconstructed_frame, "val")
        except Exception as e:
            print(f"Error during validation image logging: {e}")

    def _get_frame_pair(self, pl_module, batch, stage):
        """Extract original and reconstructed frame pair."""
        # Handle different dataset formats
        if isinstance(batch, dict):
            if "yuv" in batch:
                batch_data = batch["yuv"]
            else:
                batch_data = batch["rgb"]
        else:
            batch_data = batch
            
        batch_data = batch_data.to(pl_module.device)
        
        # Get first frame of first sequence
        frame = batch_data[0, 0, ...]  # (C, H, W)
        
        # Get reconstruction
        pl_module.eval()
        with pl_module.torch.no_grad():
            if hasattr(pl_module, 'P_frame_model'):
                # Video trainer - use i_frame_model for logging
                results = pl_module.i_frame_model(frame.unsqueeze(0), qp=32)
            else:
                # Image trainer
                results = pl_module.i_frame_model(frame.unsqueeze(0), qp=32)
            
            reconstructed_frame = results["dpb"]["frame"][0]
        
        if stage == "train":
            pl_module.train()
            
        return frame, reconstructed_frame

    def _log_images(self, trainer, original, reconstructed, prefix):
        """Log images to TensorBoard."""
        if not isinstance(trainer.logger, TensorBoardLogger):
            return
            
        try:
            # Convert YUV to RGB for visualization
            original_rgb = ycbcr2rgb(original.unsqueeze(0)).squeeze(0)
            reconstructed_rgb = ycbcr2rgb(reconstructed.unsqueeze(0)).squeeze(0)
            
            # Prepare for logging
            original_prep = prepare_image_for_logging(original_rgb)
            reconstructed_prep = prepare_image_for_logging(reconstructed_rgb)
            
            tb_writer = trainer.logger.experiment
            if original_prep is not None:
                tb_writer.add_image(f"{prefix}/Original", original_prep, trainer.global_step)
            if reconstructed_prep is not None:
                tb_writer.add_image(f"{prefix}/Reconstructed", reconstructed_prep, trainer.global_step)
                
        except Exception as e:
            print(f"Error logging images to TensorBoard: {e}")