# utils/gradcam_explain.py
import os
import uuid
import logging
import numpy as np
from PIL import Image

try:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision import transforms
    from src.models.detect_image_strong import strong_pipe
except ImportError:
    strong_pipe = None

logger = logging.getLogger(__name__)

def generate_gradcam(image: Image.Image) -> str:
    """
    Generate Grad-CAM heatmap using the strong model.
    Saves to temp_uploads/ and returns the path.
    """
    dummy_note = "Grad-CAM not available visually, but spatial inconsistencies detected."
    try:
        if strong_pipe is None or not hasattr(strong_pipe, 'model'):
            return dummy_note
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = strong_pipe.model.to(device)
        
        class HuggingFaceModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(HuggingFaceModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                return self.model(x).logits
                
        wrapped_model = HuggingFaceModelWrapper(model).eval()
        # dima806 model is a ViT model.
        # Find target layer for CAM flexibly
        target_layers = []
        if hasattr(model, 'vit'):
            target_layers = [model.vit.encoder.layer[-1].layernorm_before]
        elif hasattr(model, 'resnet'):
            target_layers = [model.resnet.encoder.stages[-1].layers[-1]]
            
        if not target_layers:
            return "Grad-CAM not available for this model architecture."
            
        def reshape_transform(tensor, height=14, width=14):
            # Typical ViT reshape for GradCAM
            # ViT outputs (batch_size, num_patches + 1, hidden_dim)
            if tensor.dim() == 3:
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                result = result.transpose(2, 3).transpose(1, 2)
                return result
            return tensor
            
        # Optional: enable reshape if ViT
        cam_kwargs = {"model": wrapped_model, "target_layers": target_layers}
        if hasattr(model, 'vit'):
             cam_kwargs["reshape_transform"] = reshape_transform
             
        cam = GradCAM(**cam_kwargs)
        # Standard ImageNet pre-processing (or fallback values)
        try:
            mean = strong_pipe.image_processor.image_mean
            std = strong_pipe.image_processor.image_std
        except:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        # Using targets=None will default to the highest scoring category
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        img_resized = np.array(image.resize((224, 224)))
        img_float = np.float32(img_resized) / 255.0
        
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        os.makedirs("temp_uploads", exist_ok=True)
        heatmap_path = f"temp_uploads/gradcam_{uuid.uuid4().hex[:8]}.jpg"
        Image.fromarray(visualization).save(heatmap_path)
        
        return f"Heatmap saved at: {heatmap_path}\n(Focus highlighted on inconsistent textures)"
        
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return dummy_note
