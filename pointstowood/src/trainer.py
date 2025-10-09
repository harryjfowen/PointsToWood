from src.dataset import create_train_loader, create_test_loader
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.statistics import calculate_harmonic_metrics, print_validation_summary, update_test_metrics_with_harmonic
from tqdm import tqdm
import numpy as np
import torch
import os
from time import sleep
from src.loss import *
from torch.optim import AdamW
import warnings
import copy
from src.pointcutmix import apply_pointcutmix_batch, recompute_edge_scores, apply_edge_aware_label_smoothing

seed = 141190
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)


def run_validation_pass(model, test_loader, device, mode_name, augmentation_mode):
    """Run a single validation pass with specified augmentation mode."""
    # Temporarily modify the dataset mode
    original_mode = test_loader.dataset.mode
    test_loader.dataset.mode = augmentation_mode
    
    test_tracker = MetricsTracker()
    
    with tqdm(total=len(test_loader), colour='cyan' if 'With' in mode_name else 'yellow', 
              ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(device)
                model_output = model(data)
                
                outputs = model_output
                
                test_tracker.update(torch.tensor(0.0), outputs, data.y)
                
                curr_metrics = test_tracker.get_averages()
                tepoch.set_description(f"Val {mode_name}")
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
    
    # Restore original mode
    test_loader.dataset.mode = original_mode
    
    return test_tracker.get_averages()


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
        model = Net(num_classes=1, C=64, num_kernel_points=32, learnable_kernels=False).to(device)  # Full EU model
        lr = 1e-3
        weight_decay = 1e-2
    else:
        from src.model import NetLight as Net
        model = Net(num_classes=1, C=16, num_kernel_points=8, learnable_kernels=True).to(device)  # Lightweight biome model
        lr = 1e-3  # Lower LR for stability
        weight_decay = 1e-2  # Higher regularization
    
    print(f'Model contains {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Using {"EU" if "eu" in args.model.lower() else "Biome"} model')

    train_loader, _ = create_train_loader(args, device)

    if args.test:
        test_loader, _ = create_test_loader(args, device)

    # Edge weighting: earlier ramp with lighter base weight
    criterion = DynamicEdgeFocalLoss(
        min_gamma=1.0,
        max_gamma=3.0,
        sharpness=2.0,
        alpha_min=0.1,      # Lighter base edge contribution
        alpha_max=1.0,      # Edges equal to semantics at end
        ramp_start=0.5,     # Begin ramping around 50% of training
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

    best_fbeta_refl, best_fbeta_no_refl, best_fbeta_harmonic = 0.0, 0.0, 0.0

    scaler = torch.amp.GradScaler(
        init_scale=256,
        growth_factor=1.01,  # Must be > 1.0 per PyTorch; ~frozen growth
        backoff_factor=0.5,
        growth_interval=10000,  # Grow very infrequently to avoid runaway scaling
        enabled=True
    )

    # EMA Loss tracking
    ema_loss = None
    ema_alpha = 0.9
    
    # EMA Model tracking with fixed decay
    ema_model = EMAModel(model, decay=0.9)
    ema_model.register()

    accumulation_steps = 1  # Disabled for debugging 
    optimizer.zero_grad(set_to_none=True)  
    accumulated_batches = 0  # Track actual accumulated batches

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")
        
        ema_model.set_epoch(epoch)  # Update EMA epoch for decay scheduling
        criterion.set_epoch(epoch, args.num_epochs)  # Update loss function epoch for edge ramping
        train_tracker = MetricsTracker()

        with tqdm(total=len(train_loader), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, data in enumerate(train_loader):
                data = data.to(device)

                # Apply PointCutMix data augmentation (only during training)
                pointcutmix_applied = False
                if hasattr(args, 'pointcutmix') and args.pointcutmix and model.training:
                    original_data = data

                    # Extract batch components for PointCutMix
                    from torch_geometric.data import Batch
                    batch_data = Batch.to_data_list(data)
                    batch_pos = [sample.pos for sample in batch_data]
                    batch_reflectance = [sample.reflectance for sample in batch_data]
                    batch_label = [sample.y for sample in batch_data]

                    # Apply PointCutMix (spatial method)
                    mixed_pos, mixed_refl, mixed_label = apply_pointcutmix_batch(
                        batch_pos, batch_reflectance, batch_label,
                        prob=getattr(args, 'pointcutmix_prob', 0.25),
                        beta=getattr(args, 'pointcutmix_beta', 1.0),
                        method='spatial'
                    )

                    # Reconstruct batch
                    from torch_geometric.data import Data
                    mixed_batch_data = []
                    for i in range(len(mixed_pos)):
                        mixed_sample = Data(
                            pos=mixed_pos[i],
                            reflectance=mixed_refl[i],
                            y=mixed_label[i],
                            sf=batch_data[i].sf,
                            edge_scores=batch_data[i].edge_scores
                        )
                        mixed_batch_data.append(mixed_sample)

                    data = Batch.from_data_list(mixed_batch_data)

                    # Check if any mixing actually occurred
                    pointcutmix_applied = not torch.equal(data.y, original_data.y)

                    # Recompute edge scores after mixing
                    if pointcutmix_applied:
                        data.edge_scores = recompute_edge_scores(data.pos, data.y)

                # Apply edge-aware label smoothing after final edge scores are computed
                if hasattr(args, 'edge_label_smoothing') and args.edge_label_smoothing and model.training:
                    data.y = apply_edge_aware_label_smoothing(
                        data.y,
                        data.edge_scores,
                        getattr(args, 'smoothing_factor', 0.1)
                    )

                # Skip batch early if any input contains NaN/Inf to prevent cascade
                try:
                    inputs_ok = True
                    for attr in ("x", "pos", "edge_scores", "y"):
                        if hasattr(data, attr):
                            tensor = getattr(data, attr)
                            if tensor is not None and not torch.isfinite(tensor).all():
                                inputs_ok = False
                                break
                    if not inputs_ok:
                        print(f"[Warning] Non-finite inputs at step {i}, skipping batch")
                        # If we've accumulated enough, force an optimizer update cycle without step
                        if accumulated_batches >= accumulation_steps:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_batches = 0
                        continue
                except Exception:
                    pass
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    model_output = model(data)

                    outputs = model_output

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

                    # Store values before clearing
                    loss_value = loss.item()
                    train_tracker.update(loss_value * accumulation_steps, outputs, data.y, data.edge_scores)

                    # Explicit memory clearing
                    del data, outputs, loss
                    torch.cuda.empty_cache()

                # Check for gradient update
                if accumulated_batches >= accumulation_steps:
                    scaler.unscale_(optimizer)

                    # Check for non-finite gradients (NaN or Inf)
                    non_finite_grads = any((p.grad is not None) and (not torch.isfinite(p.grad).all()) for p in model.parameters())
                    if non_finite_grads:
                        print(f"[Warning] Non-finite gradients at step {i}, skipping update")
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    did_step = False
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                        did_step = True
                        
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
                    
                    # Update EMA weights only if optimizer step actually happened
                    if did_step:
                        ema_model.update()

                # Update EMA loss
                current_loss = loss_value * accumulation_steps
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss

                # Metrics already updated above before memory clearing

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

        # Print kernel metrics after epoch
        kernel_metrics = []
        for module in model.modules():
            if hasattr(module, 'kernel_entropy') and hasattr(module, 'active_kernels'):
                kernel_metrics.append((module.kernel_entropy.item(), module.active_kernels.item()))

        if kernel_metrics:
            avg_entropy = sum(m[0] for m in kernel_metrics) / len(kernel_metrics)
            avg_active = sum(m[1] for m in kernel_metrics) / len(kernel_metrics)
            max_active = max(m[1] for m in kernel_metrics)
            print(f"Kernels: {avg_active:.1f} avg, {max_active:.1f} max | Entropy: {avg_entropy:.2f}")

        if args.test:
            model.eval()
            
            # Apply EMA weights for testing
            ema_model.apply_shadow()
            
            sleep(0.1)
            
            # Run validation with reflectance
            test_metrics_with_refl = run_validation_pass(model, test_loader, device, "With Reflectance", "val_with_reflectance")
            
            sleep(0.1)
            
            # Run validation without reflectance  
            test_metrics_no_refl = run_validation_pass(model, test_loader, device, "No Reflectance", "val_no_reflectance")
            
            # Calculate harmonic metrics and gaps
            harmonic_metrics = calculate_harmonic_metrics(test_metrics_with_refl, test_metrics_no_refl)
            
            # Print formatted validation summary
            print_validation_summary(epoch, harmonic_metrics)
            
            # (Removed verbose kernel debugging prints)
            
            # Update test metrics with harmonic data for logging
            test_metrics = update_test_metrics_with_harmonic(test_metrics_with_refl.copy(), harmonic_metrics)
            
            # Restore original weights for training
            ema_model.restore()
        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)

        if epoch in args.checkpoints:
            manager.save_checkpoints(args, epoch)
        
        # New early stopping: monitor harmonic Fbeta after 60% of training
        if getattr(args, 'early_stop', False) and getattr(args, 'test', False):
            min_delta = 0.001
            patience = 10
            start_epoch = max(1, int(args.num_epochs * 0.60))
            if epoch >= start_epoch:
                # Lazily init trackers
                if not hasattr(SemanticTraining, 'best_harmonic'):
                    SemanticTraining.best_harmonic = -float('inf')
                    SemanticTraining.bad_epochs = 0
                current_harm = harmonic_metrics['harmonic_fbeta'] if 'harmonic_fbeta' in harmonic_metrics else None
                if current_harm is not None:
                    if current_harm > SemanticTraining.best_harmonic + min_delta:
                        SemanticTraining.best_harmonic = current_harm
                        SemanticTraining.bad_epochs = 0
                    else:
                        SemanticTraining.bad_epochs += 1
                    if SemanticTraining.bad_epochs >= patience:
                        print(f"\nEarly stopping at epoch {epoch}: no harmonic Fbeta improvement ≥ {min_delta} for {patience} epochs (best={SemanticTraining.best_harmonic:.3f})")
                        best_epoch, best_acc = history_logger.get_best_epoch()
                        print(f"Best accuracy was {best_acc:.4f} at epoch {best_epoch}")
                        break

        # Save best models earlier in training in case edge ramp degrades later
        if args.test and epoch > int(args.num_epochs*0.25):
            # Save three fbeta models based on different validation strategies
            best_fbeta_refl = manager.save_best_model(harmonic_metrics['fbeta_with_refl'], best_fbeta_refl, os.path.join(args.wdir,'model','fbeta-' + os.path.basename(args.model)))
            best_fbeta_no_refl = manager.save_best_model(harmonic_metrics['fbeta_no_refl'], best_fbeta_no_refl, os.path.join(args.wdir,'model','fbeta-xyz-' + os.path.basename(args.model)))
            best_fbeta_harmonic = manager.save_best_model(harmonic_metrics['harmonic_fbeta'], best_fbeta_harmonic, os.path.join(args.wdir,'model','fbeta-harmonic-' + os.path.basename(args.model)))

        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.wdir,'model',args.model))
