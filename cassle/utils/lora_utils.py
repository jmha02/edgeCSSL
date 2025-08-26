"""
LoRA utilities for applying Low-Rank Adaptation to Vision Transformers
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, Optional
import timm


def create_lora_vit_encoder(
    model_name: str = 'vit_small_patch16_224',
    pretrained: bool = False,
    num_classes: int = 0,
    img_size: int = 224,
    patch_size: int = 16,
    global_pool: str = '',
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    **kwargs
) -> nn.Module:
    """
    Create a ViT encoder with LoRA adaptation applied.
    
    Args:
        model_name: Name of the timm model to use
        pretrained: Whether to use pretrained weights
        num_classes: Number of classes (0 for feature extraction)
        img_size: Input image size
        patch_size: Patch size for ViT
        global_pool: Global pooling type
        lora_r: LoRA rank (lower = more efficient)
        lora_alpha: LoRA scaling parameter
        lora_dropout: LoRA dropout rate
        target_modules: List of modules to apply LoRA to. If None, uses default ViT modules
        **kwargs: Additional arguments for timm.create_model
    
    Returns:
        LoRA-adapted ViT model
    """
    
    # Create the base ViT model using timm
    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        global_pool=global_pool,
        **kwargs
    )
    
    # Default target modules for ViT (attention and MLP layers)
    if target_modules is None:
        target_modules = [
            "attn.qkv",     # Query, Key, Value projection in attention blocks
            "attn.proj",    # Output projection in attention blocks
            "mlp.fc1",      # First FC layer in MLP blocks
            "mlp.fc2",      # Second FC layer in MLP blocks
        ]
    
    # Configure LoRA  
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # Don't adapt bias terms
        use_rslora=False,  # Use standard LoRA
        modules_to_save=None,
    )
    
    # Apply LoRA to the model
    lora_model = get_peft_model(base_model, lora_config)
    
    # Store original embed_dim for compatibility
    lora_model.embed_dim = base_model.embed_dim if hasattr(base_model, 'embed_dim') else base_model.num_features
    
    return lora_model


def get_lora_trainable_params(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about LoRA trainable parameters.
    
    Args:
        model: LoRA-adapted model
        
    Returns:
        Dictionary with parameter statistics
    """
    if hasattr(model, 'print_trainable_parameters'):
        # Get trainable parameters info
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            'trainable_params': trainable_params,
            'all_params': all_param,
            'percentage': 100 * trainable_params / all_param,
        }
    else:
        # Fallback for non-PEFT models
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        
        return {
            'trainable_params': trainable_params,
            'all_params': all_params,
            'percentage': 100 * trainable_params / all_params,
        }


def print_lora_info(model: nn.Module, model_name: str = "Model"):
    """Print LoRA adaptation information."""
    info = get_lora_trainable_params(model)
    
    print(f"\n{model_name} LoRA Configuration:")
    print(f"  Trainable params: {info['trainable_params']:,}")
    print(f"  All params: {info['all_params']:,}")
    print(f"  Trainable: {info['percentage']:.2f}%")
    
    if hasattr(model, 'peft_config'):
        config = list(model.peft_config.values())[0] if model.peft_config else None
        if config:
            print(f"  LoRA rank (r): {config.r}")
            print(f"  LoRA alpha: {config.lora_alpha}")
            print(f"  LoRA dropout: {config.lora_dropout}")
            print(f"  Target modules: {config.target_modules}")


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.
    This is typically handled automatically by PEFT, but can be called explicitly.
    """
    if hasattr(model, 'base_model'):
        # For PEFT models, only LoRA parameters should be trainable
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # For regular models, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False