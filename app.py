"""
AI SpillGuard Pro - Enterprise Oil Spill Detection System
==========================================================
Production-ready Streamlit application for real-time oil spill detection
from satellite SAR imagery using Deep Learning (U-Net + ResNet34).

Author: AI SpillGuard Team
Version: 1.0.0
License: MIT
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import os
import json
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Union, List
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CUSTOM CSS FOR ENTERPRISE UI ====================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main title styling with gradient */
    .main-title {
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a8f 100%);
        color: white;
        text-align: center;
        padding: 1.5rem 0;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    /* Enhanced alert cards with icons */
    .alert-card {
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        border-left: 5px solid #990000;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        border-left: 5px solid #cc6600;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #00cc66 0%, #009944 100%);
        color: white;
        border-left: 5px solid #006633;
    }
    
    /* Enhanced metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        padding: 0.8rem;
        transition: all 0.3s ease;
        border: 1px solid #dee2e6;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        border-color: #1f77b4;
    }
    
    /* Sidebar professional styling - Dark Mode */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a8f 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(31, 119, 180, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0d5a8f 0%, #094567 100%);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.5);
        transform: translateY(-2px);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #1e7e34 0%, #155724 100%);
        transform: translateY(-2px);
    }
    
    /* Container styling */
    .result-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Class legend items with improved spacing */
    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
        padding: 6px 8px;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }
    
    .legend-item:hover {
        background-color: rgba(31, 119, 180, 0.05);
    }
    
    .legend-color-box {
        width: 24px;
        height: 24px;
        margin-right: 12px;
        border: 2px solid #dee2e6;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling - Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #333 !important;
        color: white !important;
    }
    
    /* Image border styling */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #dee2e6, transparent);
    }
</style>
"""

class OilSpillDetector:
    """
    Enterprise-grade Backend Inference Engine for Oil Spill Detection
    
    Handles model loading, preprocessing, inference, and post-processing
    with robust error handling and performance optimizations.
    """
    
    def __init__(self, model_path: str = "best_model.pth", img_size: int = 256):
        """
        Initialize the oil spill detector.
        
        Args:
            model_path: Path to the trained model checkpoint
            img_size: Input image size for the model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.img_size = img_size
        self.num_classes = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()  # Enable mixed precision on GPU
        
        # Class configuration (4-class segmentation)
        self.class_names = ["Background", "Oil Spill", "Look-alike", "Ship/Wake"]
        
        # RGB color map for visualization (strictly RGB format)
        self.class_colors = np.array([
            [0,   0,   0  ],    # Background: Black
            [255, 0,   124],    # Oil Spill: Magenta/Pink
            [255, 204, 51 ],    # Look-alike: Yellow
            [51,  221, 255],    # Ship/Wake: Cyan
        ], dtype=np.uint8)
        
        # Load model with comprehensive error handling
        try:
            self.model = self._load_model(model_path)
            logger.info(f"Model loaded successfully on {self.device}")
            st.success(f"✅ Model loaded successfully on **{self.device}** (Mixed Precision: {self.use_amp})")
        except FileNotFoundError as e:
            error_msg = f"❌ Model file '{model_path}' not found. Please ensure the model file exists in the project directory."
            logger.error(error_msg)
            st.error(error_msg)
            st.info("💡 **Tip**: Check if 'best_model.pth' exists in the root directory or 'checkpoints/' folder.")
            raise
        except RuntimeError as e:
            error_msg = f"❌ Failed to load model architecture: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"❌ Unexpected error during model initialization: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load trained U-Net model with robust error handling.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded PyTorch model in evaluation mode
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model state loading fails
        """
        # Check file existence
        if not os.path.exists(checkpoint_path):
            # Try alternative path
            alt_path = os.path.join("checkpoints", "best_model.pth")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                logger.info(f"Using alternative model path: {alt_path}")
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path} or {alt_path}")
        
        # Initialize model architecture
        try:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=self.num_classes,
                activation=None  # Raw logits, apply softmax during inference
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model architecture: {str(e)}")
        
        # Load checkpoint with error handling
        try:
            ckpt = torch.load(
                checkpoint_path, 
                map_location=self.device,
                weights_only=False  # Allow full model loading
            )
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"Loaded model from dictionary checkpoint (epoch: {ckpt.get('epoch', 'N/A')})")
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
                logger.info("Loaded model from direct state dict")
                
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Model successfully loaded and moved to {self.device}")
            return model
            
        except KeyError as e:
            raise RuntimeError(f"Model state dict key error: {str(e)}. The checkpoint may be incompatible.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
    
    
    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray) -> Dict:
        """
        Main inference pipeline with optimized processing.
        
        Args:
            image_rgb: Input image as numpy array (H, W, 3) in RGB format, uint8
        
        Returns:
            Dictionary containing:
                - mask: Predicted segmentation mask at original size (H, W)
                - mask_model_size: Predicted mask at model input size
                - image: Original input image (RGB)
                - statistics: Per-class pixel statistics
                - mask_rgb: RGB visualization of the mask
                
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If inference fails
        """
        try:
            # Validate input
            if not isinstance(image_rgb, np.ndarray):
                raise ValueError("Input must be a numpy array")
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                raise ValueError(f"Input must be (H, W, 3), got shape {image_rgb.shape}")
            if image_rgb.dtype != np.uint8:
                logger.warning(f"Input dtype is {image_rgb.dtype}, converting to uint8")
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            
            # Store original dimensions for proper resizing
            orig_height, orig_width = image_rgb.shape[:2]
            logger.info(f"Processing image of size: {orig_width}x{orig_height}")
            
            # Preprocess: Resize to model input size (maintain RGB format)
            img_resized = cv2.resize(
                image_rgb, 
                (self.img_size, self.img_size), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize to [0, 1] and convert to tensor (C, H, W)
            x = (img_resized.astype(np.float32) / 255.0)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            
            # Inference with automatic mixed precision for GPU performance
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(x)
            
            # Post-process: Get class predictions
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Resize mask back to original image dimensions for accurate overlay
            pred_mask_original_size = cv2.resize(
                pred_mask, 
                (orig_width, orig_height), 
                interpolation=cv2.INTER_NEAREST  # Use nearest neighbor for class labels
            )
            
            # Compute statistics on model-size mask (more consistent)
            statistics = self._compute_statistics(pred_mask)
            
            # Generate RGB visualization at original size
            mask_rgb = self._decode_mask_to_rgb(pred_mask_original_size)
            
            # Build results dictionary
            results = {
                "mask": pred_mask_original_size,      # Original size for overlay
                "mask_model_size": pred_mask,         # Model size for analysis
                "image": image_rgb,                   # Original RGB image
                "statistics": statistics,             # Per-class statistics
                "mask_rgb": mask_rgb,                 # RGB visualization
                "original_size": (orig_height, orig_width),
                "model_size": (self.img_size, self.img_size)
            }
            
            logger.info(f"Inference completed successfully. Oil Spill: {statistics['Oil Spill']['percentage']:.2f}%")
            return results
            
        except ValueError as e:
            error_msg = f"Input validation error: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            raise
        except RuntimeError as e:
            error_msg = f"Inference runtime error: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during inference: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            raise
    
    def _compute_statistics(self, mask: np.ndarray) -> Dict:
        """
        Compute per-class pixel statistics using vectorized operations.
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            
        Returns:
            Dictionary with statistics for each class
        """
        total_pixels = mask.size
        stats = {}
        
        # Vectorized computation using numpy unique
        unique_classes, counts = np.unique(mask, return_counts=True)
        count_dict = dict(zip(unique_classes, counts))
        
        # Build statistics for all classes
        for cls_id in range(self.num_classes):
            count = count_dict.get(cls_id, 0)
            percentage = (count / total_pixels) * 100.0
            
            stats[self.class_names[cls_id]] = {
                "count": int(count),
                "percentage": float(percentage)
            }
        
        return stats
    
    def _decode_mask_to_rgb(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        Convert class indices to RGB color visualization.
        
        Args:
            mask_2d: 2D array of class indices (H, W)
            
        Returns:
            RGB image (H, W, 3) uint8
        """
        # Ensure valid class indices
        mask_2d = np.clip(mask_2d, 0, self.num_classes - 1).astype(np.int32)
        
        # Map class indices to RGB colors using fancy indexing
        mask_rgb = self.class_colors[mask_2d]
        
        return mask_rgb.astype(np.uint8)
    
    
    def create_overlay(
        self, 
        image_rgb: np.ndarray, 
        mask_2d: np.ndarray, 
        alpha: float = 0.5, 
        enabled_classes: Optional[List[int]] = None, 
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Create alpha-blended overlay visualization using optimized vectorized operations.
        
        Args:
            image_rgb: Original image (H, W, 3) uint8, RGB format
            mask_2d: Segmentation mask (H, W) with class indices
            alpha: Alpha blending factor (0=original, 1=full mask)
            enabled_classes: List of class IDs to display (None = all)
            draw_contours: Whether to draw white contours around segments
            
        Returns:
            Overlay image (H, W, 3) uint8, RGB format
            
        Raises:
            ValueError: If image and mask dimensions don't match
        """
        try:
            # Validate dimensions
            if image_rgb.shape[:2] != mask_2d.shape[:2]:
                raise ValueError(
                    f"Image size {image_rgb.shape[:2]} doesn't match mask size {mask_2d.shape[:2]}"
                )
            
            # Ensure uint8 format for consistent processing
            if image_rgb.dtype != np.uint8:
                img_u8 = np.clip(image_rgb * 255.0 if image_rgb.max() <= 1.0 else image_rgb, 0, 255).astype(np.uint8)
            else:
                img_u8 = image_rgb.copy()
            
            # Generate RGB mask
            mask_rgb = self._decode_mask_to_rgb(mask_2d)
            
            # Initialize output with original image
            out = img_u8.copy()
            
            # Handle class filtering
            if enabled_classes is None:
                enabled_classes = list(range(self.num_classes))
            
            # Filter out background (class 0) for better visualization
            enabled_non_bg = [cls_id for cls_id in enabled_classes if cls_id != 0]
            
            if not enabled_non_bg:
                # No classes to overlay, return original
                return out
            
            # ===== VECTORIZED ALPHA BLENDING =====
            # Create binary mask for all enabled classes at once
            combined_mask = np.isin(mask_2d, enabled_non_bg)
            
            # Vectorized alpha blending using numpy where
            # out[mask] = alpha * mask_rgb[mask] + (1 - alpha) * img_u8[mask]
            out = np.where(
                combined_mask[..., None],  # Broadcast to (H, W, 1) -> (H, W, 3)
                (alpha * mask_rgb + (1 - alpha) * img_u8).astype(np.uint8),
                img_u8
            )
            
            # ===== CONTOUR DRAWING =====
            if draw_contours:
                for cls_id in enabled_non_bg:
                    # Create binary mask for this specific class
                    class_mask = (mask_2d == cls_id).astype(np.uint8) * 255
                    
                    # Skip if no pixels of this class
                    if class_mask.sum() == 0:
                        continue
                    
                    # Find contours using OpenCV
                    contours, _ = cv2.findContours(
                        class_mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Draw white contours with slight thickness
                    cv2.drawContours(out, contours, -1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            
            return out
            
        except ValueError as e:
            error_msg = f"Overlay creation error: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error creating overlay: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            raise


class AlertSystem:
    """
    Intelligent Alert Monitoring System for Oil Spill Detection
    
    Monitors detection statistics and triggers alerts when thresholds are exceeded.
    Prioritizes Oil Spill detection as the primary alert trigger.
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize alert system with configurable thresholds.
        
        Args:
            thresholds: Dictionary mapping class names to percentage thresholds
        """
        self.thresholds = thresholds or {
            "Oil Spill": 5.0,      # Primary alert: Oil spill detection
            "Look-alike": 10.0,    # Secondary: False positives
            "Ship/Wake": 3.0       # Secondary: Ship activity
        }
        logger.info(f"AlertSystem initialized with thresholds: {self.thresholds}")
    
    def check_alerts(self, statistics: Dict) -> List[Dict]:
        """
        Check detection statistics against configured thresholds.
        
        Args:
            statistics: Per-class statistics from detection results
            
        Returns:
            List of alert dictionaries, prioritized by severity
        """
        alerts = []
        
        # PRIMARY ALERT: Oil Spill Detection (Critical Priority)
        if "Oil Spill" in statistics:
            oil_spill_stats = statistics["Oil Spill"]
            pct = oil_spill_stats["percentage"]
            threshold = self.thresholds.get("Oil Spill", 5.0)
            
            if pct > threshold:
                severity = "CRITICAL" if pct > (threshold * 2) else "WARNING"
                alerts.append({
                    "class": "Oil Spill",
                    "percentage": float(pct),
                    "threshold": float(threshold),
                    "severity": severity,
                    "priority": 1,  # Highest priority
                    "message": f"🛢️ Oil spill detected at {pct:.2f}% coverage (Threshold: {threshold}%)"
                })
                logger.warning(f"OIL SPILL ALERT: {pct:.2f}% coverage (threshold: {threshold}%)")
        
        # SECONDARY ALERTS: Other classes
        for cls_name, threshold in self.thresholds.items():
            if cls_name == "Oil Spill":
                continue  # Already handled above
                
            if cls_name in statistics:
                pct = statistics[cls_name]["percentage"]
                
                if pct > threshold:
                    severity = "CRITICAL" if pct > (threshold * 2) else "WARNING"
                    
                    # Determine icon and priority
                    if cls_name == "Look-alike":
                        icon = "⚠️"
                        priority = 2
                    elif cls_name == "Ship/Wake":
                        icon = "🚢"
                        priority = 3
                    else:
                        icon = "ℹ️"
                        priority = 4
                    
                    alerts.append({
                        "class": cls_name,
                        "percentage": float(pct),
                        "threshold": float(threshold),
                        "severity": severity,
                        "priority": priority,
                        "message": f"{icon} {cls_name} detected at {pct:.2f}% (Threshold: {threshold}%)"
                    })
                    logger.info(f"Secondary alert: {cls_name} at {pct:.2f}%")
        
        # Sort alerts by priority (lower number = higher priority)
        alerts.sort(key=lambda x: (x["priority"], -x["percentage"]))
        
        return alerts
    
    def update_threshold(self, class_name: str, new_threshold: float) -> bool:
        """
        Update alert threshold for a specific class.
        
        Args:
            class_name: Name of the class to update
            new_threshold: New threshold percentage value
            
        Returns:
            True if update successful, False otherwise
        """
        if class_name in self.thresholds:
            old_threshold = self.thresholds[class_name]
            self.thresholds[class_name] = float(new_threshold)
            logger.info(f"Updated threshold for '{class_name}': {old_threshold}% -> {new_threshold}%")
            return True
        else:
            logger.warning(f"Attempted to update threshold for unknown class: {class_name}")
            return False
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current threshold configuration."""
        return self.thresholds.copy()
    
    def render_alert(self, alert: Dict) -> None:
        """
        Render a single alert using Streamlit components.
        
        Args:
            alert: Alert dictionary with class, percentage, severity info
        """
        severity_class = "alert-critical" if alert["severity"] == "CRITICAL" else "alert-warning"
        
        st.markdown(
            f'<div class="alert-card {severity_class}">'
            f'<strong>🚨 {alert["severity"]}</strong>: {alert["message"]}'
            f'</div>',
            unsafe_allow_html=True
        )


class StorageManager:
    """
    Persistent Storage System for Detection History and Results
    
    Handles saving detection results, images, and maintaining a searchable history.
    Cross-platform compatible with robust error handling.
    """
    
    def __init__(self, storage_dir: str = "detection_history"):
        """
        Initialize storage manager.
        
        Args:
            storage_dir: Directory path for storing results
        """
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "history.json")
        
        # Create storage directory with error handling
        try:
            os.makedirs(storage_dir, exist_ok=True)
            logger.info(f"Storage directory initialized: {storage_dir}")
        except PermissionError:
            logger.error(f"Permission denied creating storage directory: {storage_dir}")
            st.error(f"❌ Permission denied: Cannot create storage directory '{storage_dir}'")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {str(e)}")
            st.error(f"❌ Failed to initialize storage: {str(e)}")
    
    def save_detection(
        self, 
        image_name: str, 
        results: Dict, 
        alerts: List[Dict], 
        overlay_img: np.ndarray, 
        mask_img: np.ndarray
    ) -> Optional[str]:
        """
        Save complete detection results with images and metadata.
        
        Args:
            image_name: Name of the input image file
            results: Detection results dictionary
            alerts: List of triggered alerts
            overlay_img: Overlay visualization (RGB, uint8)
            mask_img: Mask visualization (RGB, uint8)
            
        Returns:
            Timestamp string if successful, None otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overlay and mask images
        try:
            overlay_path = os.path.join(self.storage_dir, f"{timestamp}_overlay.png")
            mask_path = os.path.join(self.storage_dir, f"{timestamp}_mask.png")
            
            # Ensure RGB format for PIL
            if overlay_img.shape[2] == 3:
                Image.fromarray(overlay_img, mode='RGB').save(overlay_path, optimize=True)
                Image.fromarray(mask_img, mode='RGB').save(mask_path, optimize=True)
                logger.info(f"Saved images: {overlay_path}, {mask_path}")
            else:
                raise ValueError(f"Invalid image shape: {overlay_img.shape}")
                
        except Exception as e:
            error_msg = f"Failed to save images: {str(e)}"
            logger.error(error_msg)
            st.warning(f"⚠️ {error_msg}")
            overlay_path = ""
            mask_path = ""
        
        # Create detection record
        record = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "image_name": image_name,
            "statistics": results["statistics"],
            "alerts": alerts,
            "alert_count": len(alerts),
            "has_critical": any(a["severity"] == "CRITICAL" for a in alerts),
            "overlay_path": overlay_path,
            "mask_path": mask_path,
            "original_size": results.get("original_size", [0, 0]),
            "model_size": results.get("model_size", [256, 256])
        }
        
        # Update history JSON with error handling
        try:
            history = self.load_history()
            history.append(record)
            
            # Write with atomic operation (write to temp, then rename)
            temp_file = self.history_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (safe on all platforms)
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            os.rename(temp_file, self.history_file)
            
            logger.info(f"Detection record saved: {timestamp}")
            return timestamp
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON encoding error: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
        except PermissionError:
            error_msg = "Permission denied writing history file"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
        except Exception as e:
            error_msg = f"Failed to save detection history: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
    
    def load_history(self) -> List[Dict]:
        """
        Load detection history from JSON file.
        
        Returns:
            List of detection records, empty list if file doesn't exist or is corrupt
        """
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
            if not isinstance(history, list):
                logger.warning("History file is not a list, resetting")
                return []
                
            logger.info(f"Loaded {len(history)} detection records")
            return history
            
        except json.JSONDecodeError as e:
            error_msg = f"Corrupted history file: {str(e)}"
            logger.error(error_msg)
            st.warning(f"⚠️ {error_msg}. Creating new history.")
            return []
        except Exception as e:
            logger.error(f"Failed to load history: {str(e)}")
            return []
    
    def export_history_csv(self) -> Optional[pd.DataFrame]:
        """
        Export detection history as a pandas DataFrame for CSV export.
        
        Returns:
            DataFrame with detection history, None if no data
        """
        history = self.load_history()
        
        if not history:
            return None
        
        try:
            data = []
            for record in history:
                data.append({
                    "Timestamp": record.get("timestamp", "N/A"),
                    "DateTime": record.get("datetime", "N/A"),
                    "Image": record.get("image_name", "N/A"),
                    "Oil Spill (%)": f"{record['statistics']['Oil Spill']['percentage']:.2f}",
                    "Look-alike (%)": f"{record['statistics']['Look-alike']['percentage']:.2f}",
                    "Ship/Wake (%)": f"{record['statistics']['Ship/Wake']['percentage']:.2f}",
                    "Background (%)": f"{record['statistics']['Background']['percentage']:.2f}",
                    "Alerts": record.get("alert_count", len(record.get("alerts", []))),
                    "Has Critical": "Yes" if record.get("has_critical", False) else "No"
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Exported {len(df)} records to DataFrame")
            return df
            
        except Exception as e:
            error_msg = f"Failed to export history to CSV: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
    
    def clear_history(self) -> bool:
        """
        Clear all detection history and stored images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove history file
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
                logger.info("History file removed")
            
            # Remove all PNG files in storage directory
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(self.storage_dir, filename)
                    os.remove(filepath)
                    logger.debug(f"Removed: {filepath}")
            
            logger.info("Detection history cleared successfully")
            return True
            
        except PermissionError:
            logger.error("Permission denied clearing history")
            st.error("❌ Permission denied: Cannot clear history")
            return False
        except Exception as e:
            error_msg = f"Failed to clear history: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return False


# ==================== FRONTEND COMPONENTS ====================

def init_session_state():
    """
    Initialize Streamlit session state with robust error handling.
    
    Ensures all required objects are properly instantiated before use.
    """
    initialization_errors = []
    
    # Initialize OilSpillDetector
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = OilSpillDetector("best_model.pth")
            logger.info("OilSpillDetector initialized successfully")
        except FileNotFoundError as e:
            error_msg = "Model file not found. Please ensure 'best_model.pth' exists in the project directory."
            initialization_errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to initialize detector: {str(e)}"
            initialization_errors.append(error_msg)
            logger.error(error_msg)
    
    # Initialize AlertSystem
    if 'alert_system' not in st.session_state:
        try:
            st.session_state.alert_system = AlertSystem()
            logger.info("AlertSystem initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize alert system: {str(e)}"
            initialization_errors.append(error_msg)
            logger.error(error_msg)
    
    # Initialize StorageManager
    if 'storage' not in st.session_state:
        try:
            st.session_state.storage = StorageManager()
            logger.info("StorageManager initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize storage: {str(e)}"
            initialization_errors.append(error_msg)
            logger.error(error_msg)
    
    # Display initialization errors and stop if critical
    if initialization_errors:
        st.error("## ❌ Initialization Errors")
        for error in initialization_errors:
            st.error(f"• {error}")
        
        st.info("""
        ### 🔧 Troubleshooting Steps:
        1. Ensure `best_model.pth` exists in the project root or `checkpoints/` folder
        2. Check file permissions for the project directory
        3. Verify all required packages are installed: `pip install -r requirements.txt`
        4. Check Python version compatibility (3.8+)
        """)
        
        st.stop()  # Stop app execution if critical errors

def render_sidebar():
    """Render sidebar controls with improved styling"""
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        st.success(f"✅ Device: {st.session_state.detector.device}")
        
        st.markdown("---")
        
        # Alert thresholds
        st.subheader("🚨 Alert Thresholds")
        thresholds = {}
        for cls_name in ["Oil Spill", "Look-alike", "Ship/Wake"]:
            thresholds[cls_name] = st.slider(
                f"{cls_name} (%)",
                0.0, 50.0,
                st.session_state.alert_system.thresholds[cls_name],
                0.5,
                key=f"threshold_{cls_name}"
            )
            st.session_state.alert_system.update_threshold(cls_name, thresholds[cls_name])
        
        st.markdown("---")
        
        # Visualization settings
        st.subheader("🎨 Visualization")
        alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.45, 0.05)
        draw_contours = st.checkbox("Draw Contours", value=True)
        
        # Class toggles
        st.subheader("🔍 Display Classes")
        enabled_classes = []
        for cls_id, cls_name in enumerate(st.session_state.detector.class_names):
            if st.checkbox(cls_name, value=(cls_id != 0), key=f"class_{cls_id}"):
                enabled_classes.append(cls_id)
        
        st.markdown("---")
        
        # Storage
        st.subheader("💾 Storage")
        save_results = st.checkbox("Auto-save Results", value=True)
        
        st.markdown("---")
        
        # Legend with improved styling
        st.subheader("🎨 Class Legend")
        for cls_id, cls_name in enumerate(st.session_state.detector.class_names):
            color = st.session_state.detector.class_colors[cls_id]
            st.markdown(
                f'<div class="legend-item">'
                f'<div class="legend-color-box" style="background-color:rgb({color[0]},{color[1]},{color[2]});"></div>'
                f'<span>{cls_name}</span></div>',
                unsafe_allow_html=True
            )
        
        return alpha, draw_contours, enabled_classes, save_results

def render_detection_tab():
    """
    Main detection interface with enterprise-grade layout and error handling.
    
    Provides a polished user experience with:
    - Image upload and validation
    - Real-time inference
    - Visual trinity: Original | Mask | Overlay
    - Alert system integration
    - Download capabilities
    """
    st.subheader("📤 Upload Satellite SAR Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Upload SAR (Synthetic Aperture Radar) or satellite imagery for oil spill detection",
        accept_multiple_files=False
    )
    
    # Get sidebar settings
    alpha, draw_contours, enabled_classes, save_results = render_sidebar()
    
    if uploaded_file is not None:
        # ===== IMAGE LOADING =====
        try:
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image_pil)
            
            # Validate image
            if image_np.size == 0:
                st.error("❌ Uploaded image is empty or corrupted")
                return
                
            logger.info(f"Loaded image: {uploaded_file.name}, size: {image_np.shape}")
            
        except Exception as e:
            st.error(f"❌ Failed to load image: {str(e)}")
            st.info("💡 Please ensure the file is a valid image format (PNG, JPG, TIFF)")
            return
        
        # ===== INFERENCE =====
        with st.spinner("🔄 Running AI detection... Please wait"):
            try:
                results = st.session_state.detector.predict(image_np)
                logger.info(f"Inference completed for {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Inference failed: {str(e)}")
                logger.error(f"Inference error: {str(e)}", exc_info=True)
                return
        
        # ===== ALERT CHECKING =====
        alerts = st.session_state.alert_system.check_alerts(results["statistics"])
        
        # Display alerts with enhanced styling
        st.markdown("---")
        if alerts:
            st.markdown("### 🚨 ALERTS TRIGGERED")
            
            for alert in alerts:
                if alert["severity"] == "CRITICAL":
                    st.error(f"**🚨 {alert['severity']}**: {alert['message']}")
                else:
                    st.warning(f"**⚠️ {alert['severity']}**: {alert['message']}")
        else:
            st.markdown(
                '<div class="alert-card alert-success">'
                '<strong>✅ ALL CLEAR</strong>: No critical alerts detected. All parameters within normal range.'
                '</div>',
                unsafe_allow_html=True
            )
        
        # ===== OVERLAY CREATION =====
        try:
            overlay = st.session_state.detector.create_overlay(
                results["image"], 
                results["mask"],
                alpha=alpha, 
                enabled_classes=enabled_classes,
                draw_contours=draw_contours
            )
        except Exception as e:
            st.error(f"❌ Failed to create overlay: {str(e)}")
            logger.error(f"Overlay error: {str(e)}", exc_info=True)
            return
        
        # ===== VISUAL TRINITY: ORIGINAL | MASK | OVERLAY =====
        st.markdown("---")
        st.markdown("### 🖼️ Detection Results - Visual Comparison")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📷 Original SAR")
                st.image(
                    results["image"], 
                    use_column_width=True, 
                    caption=f"Original Image ({results['original_size'][1]}×{results['original_size'][0]})"
                )
            
            with col2:
                st.markdown("#### 🎯 Categorical Mask")
                st.image(
                    results["mask_rgb"], 
                    use_column_width=True, 
                    caption="Segmentation Mask (4-Class)"
                )
            
            with col3:
                st.markdown("#### 🔍 Alpha-Blended Overlay")
                st.image(
                    overlay, 
                    use_column_width=True, 
                    caption=f"Detection Overlay (α={alpha:.2f})"
                )
        
        # ===== STATISTICS DASHBOARD =====
        st.markdown("---")
        st.markdown("### 📊 Pixel-Level Detection Statistics")
        
        with st.container():
            stat_cols = st.columns(4)
            
            for idx, (cls_name, stat) in enumerate(results["statistics"].items()):
                with stat_cols[idx]:
                    # Determine metric styling
                    if cls_name == "Background":
                        delta_color = "off"
                        emoji = "⬛"
                    elif cls_name == "Oil Spill":
                        delta_color = "normal" if stat["percentage"] > 0 else "off"
                        emoji = "🛢️"
                    elif cls_name == "Look-alike":
                        delta_color = "normal" if stat["percentage"] > 0 else "off"
                        emoji = "⚠️"
                    else:  # Ship/Wake
                        delta_color = "normal" if stat["percentage"] > 0 else "off"
                        emoji = "🚢"
                    
                    st.metric(
                        label=f"{emoji} {cls_name}",
                        value=f"{stat['percentage']:.2f}%",
                        delta=f"{stat['count']:,} px",
                        delta_color=delta_color
                    )
        
        # ===== TECHNICAL METADATA =====
        with st.expander("🔬 Technical Metadata", expanded=False):
            meta_col1, meta_col2 = st.columns(2)
            
            with meta_col1:
                st.markdown(f"""
                **Image Information:**
                - Filename: `{uploaded_file.name}`
                - Original Size: {results['original_size'][1]} × {results['original_size'][0]} px
                - Model Input Size: {results['model_size'][0]} × {results['model_size'][1]} px
                - Color Space: RGB
                """)
            
            with meta_col2:
                st.markdown(f"""
                **Detection Configuration:**
                - Model: U-Net + ResNet34
                - Device: {st.session_state.detector.device}
                - Mixed Precision: {st.session_state.detector.use_amp}
                - Overlay Alpha: {alpha:.2f}
                - Contours: {'Enabled' if draw_contours else 'Disabled'}
                """)
        
        # ===== AUTO-SAVE =====
        if save_results:
            with st.spinner("💾 Saving detection results..."):
                try:
                    timestamp = st.session_state.storage.save_detection(
                        uploaded_file.name,
                        results,
                        alerts,
                        overlay,
                        results["mask_rgb"]
                    )
                    
                    if timestamp:
                        st.success(f"✅ Results automatically saved: `{timestamp}`")
                    else:
                        st.warning("⚠️ Auto-save failed, but you can still download results manually")
                        
                except Exception as e:
                    st.warning(f"⚠️ Auto-save error: {str(e)}")
                    logger.error(f"Save error: {str(e)}", exc_info=True)
        
        # ===== DOWNLOAD SECTION =====
        st.markdown("---")
        st.markdown("### ⬇️ Export & Download Results")
        
        with st.container():
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                buf = io.BytesIO()
                Image.fromarray(overlay).save(buf, format="PNG", optimize=True)
                st.download_button(
                    "📥 Download Overlay",
                    buf.getvalue(),
                    f"{uploaded_file.name.split('.')[0]}_overlay.png",
                    "image/png",
                    use_container_width=True,
                    help="Download alpha-blended overlay visualization"
                )
            
            with col_dl2:
                buf2 = io.BytesIO()
                Image.fromarray(results["mask_rgb"]).save(buf2, format="PNG", optimize=True)
                st.download_button(
                    "📥 Download Mask",
                    buf2.getvalue(),
                    f"{uploaded_file.name.split('.')[0]}_mask.png",
                    "image/png",
                    use_container_width=True,
                    help="Download categorical segmentation mask"
                )
            
            with col_dl3:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "image": uploaded_file.name,
                    "statistics": results["statistics"],
                    "alerts": alerts,
                    "configuration": {
                        "model": "U-Net ResNet34",
                        "device": str(st.session_state.detector.device),
                        "alpha": alpha,
                        "enabled_classes": enabled_classes
                    }
                }
                st.download_button(
                    "📥 Download Report (JSON)",
                    json.dumps(report, indent=2),
                    f"{uploaded_file.name.split('.')[0]}_report.json",
                    "application/json",
                    use_container_width=True,
                    help="Download complete detection report"
                )
    
    else:
        # ===== UPLOAD PROMPT =====
        st.info("👆 Upload a satellite or SAR image to begin oil spill detection")
        
        # System information
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        
        with st.container():
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown(f"""
                **Model Architecture:**
                - Type: U-Net
                - Encoder: ResNet34
                - Classes: {len(st.session_state.detector.class_names)}
                - Input Size: {st.session_state.detector.img_size}×{st.session_state.detector.img_size}
                """)
            
            with info_col2:
                st.markdown(f"""
                **Runtime Environment:**
                - Device: {st.session_state.detector.device}
                - Mixed Precision: {st.session_state.detector.use_amp}
                - Supported Formats: PNG, JPG, JPEG, TIFF, BMP
                """)
        
        # Sample usage guide
        with st.expander("📖 Usage Guide", expanded=False):
            st.markdown("""
            **How to use AI SpillGuard Pro:**
            
            1. **Upload Image**: Click the upload button and select a SAR or satellite image
            2. **Wait for Processing**: The AI model will process your image (typically < 2 seconds)
            3. **Review Results**: Examine the visual comparison and detection statistics
            4. **Check Alerts**: Any oil spills exceeding thresholds will trigger alerts
            5. **Download**: Export overlays, masks, or JSON reports for further analysis
            6. **Configure**: Adjust thresholds and visualization settings in the sidebar
            
            **Supported Image Types:**
            - SAR (Synthetic Aperture Radar) imagery
            - Optical satellite imagery
            - Multispectral data (converted to RGB)
            
            **Detection Classes:**
            - 🛢️ **Oil Spill**: Primary target for detection
            - ⚠️ **Look-alike**: Natural phenomena that resemble oil spills
            - 🚢 **Ship/Wake**: Vessel activity and wake patterns
            - ⬛ **Background**: Clean ocean surface
            """)

def render_history_tab():
    """
    Detection History Viewer with Enterprise-grade Layout
    
    Displays past detection results with export capabilities and statistics.
    """
    st.subheader("📜 Detection History & Analytics")
    
    try:
        df = st.session_state.storage.export_history_csv()
    except Exception as e:
        st.error(f"❌ Failed to load history: {str(e)}")
        logger.error(f"History load error: {str(e)}", exc_info=True)
        return
    
    if df is not None and not df.empty:
        # Display history table in styled container
        with st.container():
            st.markdown("#### 📊 Detection Records")
            st.dataframe(
                df, 
                use_container_width=True, 
                height=400,
                hide_index=True
            )
        
        # Export and management options
        st.markdown("---")
        st.markdown("### 📤 Management & Export")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Export as CSV",
                    csv_data,
                    "spill_detection_history.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Download complete history as CSV file"
                )
            
            with col2:
                json_data = json.dumps(
                    st.session_state.storage.load_history(), 
                    indent=2
                )
                st.download_button(
                    "📥 Export as JSON",
                    json_data,
                    "spill_detection_history.json",
                    "application/json",
                    use_container_width=True,
                    help="Download complete history with metadata"
                )
            
            with col3:
                if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                    if st.session_state.storage.clear_history():
                        st.success("✅ History cleared successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear history")
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📈 Historical Analytics")
        
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            total_detections = len(df)
            total_alerts = df["Alerts"].sum()
            critical_count = (df["Has Critical"] == "Yes").sum()
            alert_rate = (total_alerts / total_detections * 100) if total_detections > 0 else 0
            
            col1.metric("Total Detections", total_detections, help="Number of images processed")
            col2.metric("Total Alerts", int(total_alerts), help="Total alert count across all detections")
            col3.metric("Critical Alerts", critical_count, help="Number of critical oil spill alerts")
            col4.metric("Alert Rate", f"{alert_rate:.1f}%", help="Percentage of detections with alerts")
        
        # Additional analytics
        with st.expander("📊 Detailed Statistics", expanded=False):
            if not df.empty:
                st.markdown("#### Oil Spill Coverage Distribution")
                
                # Convert percentage strings to floats for analysis
                try:
                    oil_spill_values = df["Oil Spill (%)"].astype(str).str.rstrip('%').astype(float)
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    stats_col1.metric("Average Coverage", f"{oil_spill_values.mean():.2f}%")
                    stats_col2.metric("Maximum Coverage", f"{oil_spill_values.max():.2f}%")
                    stats_col3.metric("Minimum Coverage", f"{oil_spill_values.min():.2f}%")
                except Exception as e:
                    st.warning(f"Could not compute statistics: {str(e)}")
    
    else:
        st.info("📭 No detection history available yet")
        st.markdown("""
        Detection history will appear here once you start processing images.
        Each detection is automatically saved with:
        - Timestamp and image information
        - Segmentation statistics
        - Alert records
        - Overlay and mask visualizations
        """)

def render_api_tab():
    """
    API Integration Documentation
    
    Provides code examples and documentation for programmatic use.
    """
    st.subheader("🔌 API Integration Guide")
    
    # Python API section
    st.markdown("### 🐍 Python API Usage")
    st.markdown("The detector can be used programmatically in your Python applications:")
    
    st.code('''from app import OilSpillDetector
import cv2
import numpy as np

# Initialize detector
detector = OilSpillDetector("best_model.pth")

# Load and preprocess image (ensure RGB format)
image = cv2.imread("satellite_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = detector.predict(image_rgb)

# Access results
print("Statistics:", results["statistics"])
print("Mask shape:", results["mask"].shape)
print("Oil Spill %:", results["statistics"]["Oil Spill"]["percentage"])

# Create visualization
overlay = detector.create_overlay(
    results["image"], 
    results["mask"], 
    alpha=0.5
)

# Save results
cv2.imwrite("overlay_output.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
''', language='python')
    
    st.markdown("---")
    
    # FastAPI section
    st.markdown("### 🌐 REST API (FastAPI Implementation)")
    st.markdown("Deploy as a production REST API service:")
    
    st.code('''from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app import OilSpillDetector, AlertSystem
import numpy as np
from PIL import Image
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI SpillGuard Pro API",
    description="Oil spill detection REST API",
    version="1.0.0"
)

# Initialize detector (once at startup)
detector = OilSpillDetector("best_model.pth")
alert_system = AlertSystem()

@app.post("/api/v1/detect", response_model=dict)
async def detect_oil_spill(file: UploadFile = File(...)):
    # Detect oil spills in uploaded satellite image
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Run inference
        results = detector.predict(image_np)
        
        # Check alerts
        alerts = alert_system.check_alerts(results["statistics"])
        
        # Build response
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "image_size": results["original_size"],
            "statistics": results["statistics"],
            "alerts": alerts,
            "alert_count": len(alerts),
            "has_oil_spill": results["statistics"]["Oil Spill"]["percentage"] > 0
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "model": "U-Net ResNet34"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''', language='python')
    
    st.markdown("---")
    
    # cURL examples
    st.markdown("### 📡 cURL Examples")
    st.markdown("**Detect oil spill in image:**")
    st.code('''curl -X POST "http://localhost:8000/api/v1/detect" \\
  -F "file=@satellite_image.png" \\
  -H "accept: application/json"
''', language='bash')
    
    st.markdown("**Health check:**")
    st.code('''curl -X GET "http://localhost:8000/api/v1/health"
''', language='bash')
    
    st.markdown("---")
    
    # Response format
    st.markdown("### 📄 Response Format")
    st.code('''{
  "status": "success",
  "filename": "satellite_image.png",
  "image_size": [512, 512],
  "statistics": {
    "Background": {
      "count": 221184,
      "percentage": 84.35
    },
    "Oil Spill": {
      "count": 28672,
      "percentage": 10.94
    },
    "Look-alike": {
      "count": 8192,
      "percentage": 3.12
    },
    "Ship/Wake": {
      "count": 4096,
      "percentage": 1.56
    }
  },
  "alerts": [
    {
      "class": "Oil Spill",
      "percentage": 10.94,
      "threshold": 5.0,
      "severity": "CRITICAL",
      "priority": 1,
      "message": "🛢️ Oil spill detected at 10.94% coverage"
    }
  ],
  "alert_count": 1,
  "has_oil_spill": true
}
''', language='json')
    
    st.markdown("---")
    
    # Docker deployment
    st.markdown("### 🐳 Docker Deployment")
    st.markdown("**Dockerfile:**")
    st.code('''FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY best_model.pth .

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
''', language='dockerfile')
    
    st.markdown("**Build and run:**")
    st.code('''docker build -t ai-spillguard-api .
docker run -p 8000:8000 ai-spillguard-api
''', language='bash')
    
    # Additional resources
    st.markdown("---")
    st.markdown("### 📚 Additional Resources")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Information:**
            - Architecture: U-Net with ResNet34 encoder
            - Input: 256×256 RGB images
            - Output: 4-class segmentation
            - Framework: PyTorch + Segmentation Models PyTorch
            """)
        
        with col2:
            st.markdown("""
            **Performance:**
            - CPU Inference: ~1-2 seconds
            - GPU Inference: ~0.1-0.3 seconds
            - Mixed Precision: Supported (GPU only)
            - Batch Processing: Not implemented (single image)
            """)

# ==================== MAIN APPLICATION ====================

def main():
    """
    Main Streamlit Application Entry Point
    
    Configures the app and renders the multi-tab interface.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI SpillGuard Pro - Oil Spill Detection",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "AI SpillGuard Pro v1.0 - Enterprise Oil Spill Detection System"
        }
    )
    
    # Inject custom CSS for enterprise styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state (detector, alert system, storage)
    init_session_state()
    
    # Main header with gradient styling
    st.markdown(
        '<h1 class="main-title">🛢️ AI SpillGuard Pro<br>'
        '<small style="font-size: 0.6em;">Real-Time Oil Spill Detection & Monitoring System</small></h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666; margin-top: -1rem; margin-bottom: 2rem;">'
        '<strong>Enterprise-grade</strong> satellite image segmentation powered by Deep Learning (U-Net + ResNet34)'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detection & Analysis", 
        "📊 History & Analytics", 
        "🔌 API Integration"
    ])
    
    with tab1:
        render_detection_tab()
    
    with tab2:
        render_history_tab()
    
    with tab3:
        render_api_tab()
    
    # Footer with system info
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.caption("🛢️ **AI SpillGuard Pro** v1.0.0")
    
    with footer_col2:
        st.caption(f"⚡ Powered by U-Net + ResNet34 | Device: {st.session_state.detector.device}")
    
    with footer_col3:
        st.caption("🔬 Real-time Inference Engine | 4-Class Segmentation")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.stop()
