from transformers.trainer_callback import TrainerCallback
import pdb
import logging

logger = logging.getLogger(__name__)

class CustomCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.model.base_model.model.current_step = state.global_step
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """called when logging is performed, force to output training loss"""
        if logs is not None and state.global_step > 0:
            loss_value = None
            if 'train_loss' in logs:
                loss_value = logs['train_loss']
            elif 'loss' in logs:
                loss_value = logs['loss']
            
            if loss_value is not None:
                print(f"Step {state.global_step}: Loss = {loss_value:.6f}")
                logger.info(f"Step {state.global_step}: Loss = {loss_value:.6f}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """called when each training step ends"""
        self.step_count += 1
        
        # force to output information for each step
        if self.step_count % args.logging_steps == 0:
            print(f"Completed step {state.global_step}")
            
        if hasattr(kwargs, 'logs') and kwargs['logs'] is not None:
            logs = kwargs['logs']
            loss_value = None
            if 'train_loss' in logs:
                loss_value = logs['train_loss']
            elif 'loss' in logs:
                loss_value = logs['loss']
                
            if loss_value is not None:
                print(f"Step {state.global_step}: Loss = {loss_value:.6f}")
                logger.info(f"Step {state.global_step}: Loss = {loss_value:.6f}")


class LossLoggingCallback(TrainerCallback):
    """callback for forcing to output loss"""
    
    def __init__(self):
        self.last_loss = None
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """force to check and output loss when each step ends"""
        self.step_count += 1
        
        # get loss from trainer's logs
        if hasattr(kwargs.get('model', None), 'trainer'):
            trainer = kwargs['model'].trainer
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                if trainer.state.log_history:
                    latest_log = trainer.state.log_history[-1]
                    if 'train_loss' in latest_log:
                        loss_value = latest_log['train_loss']
                        print(f"[FORCE LOG] Step {state.global_step}: Loss = {loss_value:.6f}")
        
        # output current status for each step
        if self.step_count % args.logging_steps == 0:
            print(f"[STATUS] Completed step {state.global_step}, Total steps: {state.max_steps}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """force to output log"""
        if logs is not None:
            for key, value in logs.items():
                if 'loss' in key.lower():
                    print(f"[LOSS LOG] Step {state.global_step}: {key} = {value:.6f}")
                    self.last_loss = value