import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.optimization import get_scheduler
from typing import List, Optional


class MiMoAudioTrainer(Trainer):
    """
    Multimodal audio trainer
    
    Extends Hugging Face Trainer to support grouped sampling by length, reducing padding and improving training efficiency.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Get training sampler
        
        Returns:
            Training sampler instance, using standard Trainer sampler
        """
        return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Create optimizer
        
        Supports different learning rates for different modules:  
        - Other modules use default learning rate
        Also supports weight decay and 8-bit Adam optimizer
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            # Get parameters that need weight decay (excluding bias and LayerNorm)
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Get learning rates from args
            input_local_lr = getattr(self.args, 'input_local_lr', self.args.learning_rate)
            output_local_lr = getattr(self.args, 'output_local_lr', self.args.learning_rate)
            default_lr = self.args.learning_rate
            
            # Categorize parameters by module type and weight decay
            input_local_params_decay = []
            input_local_params_no_decay = []
            output_local_params_decay = []
            output_local_params_no_decay = []
            other_params_decay = []
            other_params_no_decay = []
            
            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Check if parameter belongs to input_local_transformer
                if "input_local_transformer" in name:
                    if name in decay_parameters:
                        input_local_params_decay.append(param)
                    else:
                        input_local_params_no_decay.append(param)
                # Check if parameter belongs to local_transformer (but not input_local_transformer)
                elif "local_transformer" in name:
                    if name in decay_parameters:
                        output_local_params_decay.append(param)
                    else:
                        output_local_params_no_decay.append(param)
                # Other parameters
                else:
                    if name in decay_parameters:
                        other_params_decay.append(param)
                    else:
                        other_params_no_decay.append(param)
            
            # Create parameter groups with different learning rates
            optimizer_grouped_parameters = []
            
            # input_local_transformer parameters
            if input_local_params_decay:
                optimizer_grouped_parameters.append({
                    "params": input_local_params_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": input_local_lr,
                })
            if input_local_params_no_decay:
                optimizer_grouped_parameters.append({
                    "params": input_local_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": input_local_lr,
                })
                
            # local_transformer parameters (output)
            if output_local_params_decay:
                optimizer_grouped_parameters.append({
                    "params": output_local_params_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": output_local_lr,
                })
            if output_local_params_no_decay:
                optimizer_grouped_parameters.append({
                    "params": output_local_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": output_local_lr,
                })
                
            # Other parameters with default learning rate
            if other_params_decay:
                optimizer_grouped_parameters.append({
                    "params": other_params_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": default_lr,
                })
            if other_params_no_decay:
                optimizer_grouped_parameters.append({
                    "params": other_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": default_lr,
                })

            # Get optimizer class and parameters
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # Remove lr from optimizer_kwargs since we set it per parameter group
            optimizer_kwargs.pop('lr', None)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # Log parameter group information
            logger.info(f"Created optimizer with {len(optimizer_grouped_parameters)} parameter groups:")
            for i, group in enumerate(optimizer_grouped_parameters):
                num_params = sum(p.numel() for p in group['params'])
                logger.info(f"  Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, params={num_params/1e6:.2f}M")
            
            # If using 8-bit Adam, need special handling for Embedding layer
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"Skipped module {module}: {skipped/2**20}M parameters")
                        # Embedding layer uses 32-bit optimization
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: using fp32 optimization for {module}")
                logger.info(f"Total skipped: {skipped/2**20}M parameters")

        return self.optimizer