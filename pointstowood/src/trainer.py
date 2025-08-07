from src.dataset import create_train_loader, create_test_loader
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from tqdm import tqdm
import numpy as np
import torch
import os
from time import sleep
from src.loss import *
from torch.optim import AdamW
import warnings
import copy

seed = 141190
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)


class EMAModel:
    """Exponential Moving Average for model weights with fixed decay."""
    
    def __init__(self, model, decay=0.9):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
    def get_current_decay(self):
        """Get current decay (fixed)."""
        return self.decay
    
    def set_epoch(self, epoch):
        """Update current epoch (kept for compatibility)."""
        pass
        
    def register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights with current decay."""
        current_decay = self.get_current_decay()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = current_decay * self.shadow[name] + (1 - current_decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def SemanticTraining(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    torch.autograd.set_detect_anomaly(True)

    # Auto-detect model type based on model name
    if 'eu' in args.model.lower():
        from src.model import NetFull as Net
        model = Net(num_classes=1, C=32).to(device)  # Full EU model
        lr = 1e-4
        weight_decay = 1e-2
    else:
        from src.model import NetLight as Net
        model = Net(num_classes=1, C=8).to(device)  # Lightweight biome model
        lr = 1e-3  # Lower LR for stability
        weight_decay = 1e-2  # Higher regularization
    
    print(f'Model contains {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Using {"EU" if "eu" in args.model.lower() else "Biome"} model')

    train_loader, _ = create_train_loader(args, device)

    if args.test:
        test_loader, _ = create_test_loader(args, device)

    criterion = DifficultyAwareFocalLoss(
        min_gamma=1.0,
        max_gamma=3.0,
        sharpness=2.0,
        alpha=0.5,
        beta=1.0,
        use_weight_scaling=True,
        reduction="mean"
    )

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr) 

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,  
        total_steps=args.num_epochs, 
        pct_start=0.10,
        anneal_strategy='cos', 
        div_factor=25  
    )
            
    manager = ModelManager(model, device)
    history_logger = HistoryLogger(args)
    wandb_logger = WandbLogger(args)

    if os.path.isfile(os.path.join(args.wdir,'model',args.model)):
        print("Loading model")
        try:
            manager.load_model(os.path.join(args.wdir,'model',args.model))
        except KeyError:
            print("Failed to load, creating new...")
            torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))
    else:
        print("\nModel not found, creating new file...")
        torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))

    best_ba_train, best_f1_train, best_ba_test, best_f1_test, best_fbeta_test, best_precision_test = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    scaler = torch.amp.GradScaler(
        init_scale=256,
        growth_factor=1.1,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )

    # EMA Loss tracking
    ema_loss = None
    ema_alpha = 0.9
    
    # EMA Model tracking with fixed decay
    ema_model = EMAModel(model, decay=0.9)
    ema_model.register()

    accumulation_steps = 6 
    optimizer.zero_grad(set_to_none=True)  
    accumulated_batches = 0  # Track actual accumulated batches

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")
        
        ema_model.set_epoch(epoch)  # Update EMA epoch for decay scheduling
        train_tracker = MetricsTracker()

        with tqdm(total=len(train_loader), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, data in enumerate(train_loader):
                data = data.to(device)
                
                with torch.autocast("cuda"):
                    outputs = model(data)

                    if torch.isnan(outputs).any():
                        print(f"[Warning] NaN in model outputs at step {i}, skipping batch")
                        # Force gradient update if we've accumulated enough
                        if accumulated_batches >= accumulation_steps:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_batches = 0
                        continue
                    
                    loss = criterion(outputs, data.y, data.edge_scores)
                    
                    # Less aggressive clamping
                    loss = torch.clamp(loss, min=0.0, max=10.0)  # Allow zero loss, lower max
                    loss = loss / accumulation_steps
                    
                    scaler.scale(loss).backward()
                    accumulated_batches += 1

                # Check for gradient update
                if accumulated_batches >= accumulation_steps:
                    scaler.unscale_(optimizer)

                    # Check for NaN gradients
                    nan_grads = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
                    if nan_grads:
                        print(f"[Warning] NaN in gradients at step {i}, skipping update")
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    try:
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Update EMA weights after successful optimizer step
                        ema_model.update()
                        
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                    except RuntimeError as e:
                        if "overflow" in str(e).lower():
                            print(f"[Overflow] AMP overflow at step {i}. Reducing scale.")
                            scaler.update()  # Let scaler handle scale reduction automatically
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_batches = 0
                        else:
                            raise e

                # Update EMA loss
                current_loss = loss * accumulation_steps
                if ema_loss is None:
                    ema_loss = current_loss.item()
                else:
                    ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss.item()

                train_tracker.update(current_loss, outputs, data.y, data.edge_scores)

                # Update progress bar with current batch metrics
                current_metrics = train_tracker.get_averages()
                tepoch.set_postfix({
                    'Lr': optimizer.param_groups[0]["lr"],
                    'Lo': round(current_metrics['loss'], 5),
                    'EMA': round(ema_loss, 5),
                    'EDecay': f'{ema_model.get_current_decay():.3f}',
                    'Scale': f'{scaler.get_scale():.0f}',
                    'Ac': round(current_metrics['accuracy'], 3),
                    'Pr': round(current_metrics['precision'], 3),
                    'Re': round(current_metrics['recall'], 3),
                    'F1': round(current_metrics['f1'], 3),
                    'mIoU': round(current_metrics['miou'], 3),
                })
                tepoch.update(1)
            tepoch.close()
            
        # Handle any remaining gradients at epoch end
        if accumulated_batches > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA weights for final step
            ema_model.update()
            
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0

        # Step the scheduler
        lr_scheduler.step()

        train_metrics = train_tracker.get_averages()

        if args.test:
            model.eval()
            
            # Apply EMA weights for testing
            ema_model.apply_shadow()
            
            sleep(0.1)
            test_tracker = MetricsTracker()

            with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        data = data.to(device)
                        outputs = model(data)
                        
                        test_tracker.update_with_smoothing(torch.tensor(0.0), outputs, data.y, data.pos, data.batch, data.edge_scores)
                        
                        curr_metrics = test_tracker.get_averages()
                        tepoch.set_description(f"Test (EMA)")
                        tepoch.update()
                        tepoch.set_postfix({
                            'Ac': np.around(curr_metrics['accuracy'], 3),
                            'Pr': np.around(curr_metrics['precision'], 3),
                            'Re': np.around(curr_metrics['recall'], 3),
                            'F1': np.around(curr_metrics['f1'], 3),
                            'Fbeta': np.around(curr_metrics['fbeta'], 3),
                            'mIoU': np.around(curr_metrics['miou'], 3),
                        })
                    tepoch.close()
            
            test_metrics = test_tracker.get_averages()
            
            # Restore original weights for training
            ema_model.restore()
        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)

        if epoch in args.checkpoints:
            manager.save_checkpoints(args, epoch)
        
        if args.stop_early:
            if epoch > 10: 
                # Use EMA loss for more stable early stopping
                current_ema = ema_loss
                if not hasattr(SemanticTraining, 'prev_ema_loss'):
                    SemanticTraining.prev_ema_loss = current_ema
                    SemanticTraining.consec_decreases = 0
                else:
                    if current_ema > SemanticTraining.prev_ema_loss:
                        SemanticTraining.consec_decreases += 1
                    else:
                        SemanticTraining.consec_decreases = 0
                SemanticTraining.prev_ema_loss = current_ema
                    
                if SemanticTraining.consec_decreases >= 10: 
                    print(f"\nStopping early at epoch {epoch} - EMA loss increased for {SemanticTraining.consec_decreases} consecutive epochs")
                    best_epoch, best_acc = history_logger.get_best_epoch()
                    print(f"Best accuracy was {best_acc:.4f} at epoch {best_epoch}")
                    break

        if epoch > int(args.num_epochs*0.10) and not args.test:
            best_ba_train = manager.save_best_model(train_metrics['accuracy'], best_ba_train, os.path.join(args.wdir,'model','ba-' + os.path.basename(args.model)))
            best_f1_train = manager.save_best_model(train_metrics['f1'], best_f1_train, os.path.join(args.wdir,'model','f1-' + os.path.basename(args.model)))
        
        if args.test and epoch > int(args.num_epochs*0.50):
            best_ba_test = manager.save_best_model(test_metrics['accuracy'], best_ba_test, os.path.join(args.wdir,'model','ba-' + os.path.basename(args.model)))
            best_f1_test = manager.save_best_model(test_metrics['f1'], best_f1_test, os.path.join(args.wdir,'model','f1-' + os.path.basename(args.model)))
            best_fbeta_test = manager.save_best_model(test_metrics['fbeta'], best_fbeta_test, os.path.join(args.wdir,'model','fbeta-' + os.path.basename(args.model)))
            best_precision_test = manager.save_best_model(test_metrics['precision'], best_precision_test, os.path.join(args.wdir,'model','precision-' + os.path.basename(args.model)))

        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.wdir,'model',args.model))
