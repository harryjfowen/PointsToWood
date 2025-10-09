import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from src.dataset import create_train_loader, create_test_loader
from src.loss import DynamicEdgeFocalLoss
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.pointcutmix import apply_pointcutmix_batch, recompute_edge_scores, apply_edge_aware_label_smoothing


def distillation_loss(student_logits, teacher_logits, targets, edge_scores, alpha=0.7, temperature=3.0, hard_loss_fn=None):
    """
    Combined distillation loss with edge-weighted hard targets.

    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs
        targets: Hard target labels
        edge_scores: Edge importance scores
        alpha: Weight for distillation loss (vs hard loss)
        temperature: Temperature for distillation softmax
        hard_loss_fn: Pre-initialized loss function
    """
    # Distillation component (soft targets from teacher) - proper binary classification
    teacher_logits_scaled = teacher_logits / temperature
    student_logits_scaled = student_logits / temperature
    teacher_probs = torch.sigmoid(teacher_logits_scaled)

    # Use binary cross entropy with logits (autocast safe)
    distill_loss = F.binary_cross_entropy_with_logits(student_logits_scaled, teacher_probs.detach())

    # Hard target component (your sophisticated loss)
    hard_loss = hard_loss_fn(student_logits, targets, edge_scores)

    # Combine losses
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    return total_loss


def SemanticDistillation(args):
    """
    Distillation training function that creates lightweight biome-specific models
    from a comprehensive EU teacher model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)

    # Load teacher model (always EU model)
    from src.model import NetFull as TeacherNet
    teacher_model = TeacherNet(num_classes=1, C=64, num_kernel_points=32, learnable_kernels=False).to(device)

    # Load teacher weights
    teacher_path = os.path.join(args.wdir, 'model', args.teacher_model)
    teacher_checkpoint = torch.load(teacher_path, map_location=device, weights_only=True)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()  # Keep in eval mode
    print(f'Loaded teacher model: {args.teacher_model}')

    # Create student model (always biome model for distillation)
    from src.model import NetLight as StudentNet
    student_model = StudentNet(num_classes=1, C=16, num_kernel_points=8, learnable_kernels=True).to(device)
    lr = args.max_lr  # Use max_lr argument
    weight_decay = args.weight_decay  # Use weight_decay argument

    print(f'Teacher model: {sum(p.numel() for p in teacher_model.parameters()):,} parameters')
    print(f'Student model: {sum(p.numel() for p in student_model.parameters()):,} parameters')
    print(f'Compression ratio: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.1f}x')

    train_loader, _ = create_train_loader(args, device)

    if args.test:
        test_loader, _ = create_test_loader(args, device)

    # Edge weighting and other training setup (same as original)
    edge_decay_gamma = 0.96
    edge_base_weight = 0.5
    optimizer = AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Same LR scheduler as train.py - OneCycleLR with warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=args.num_epochs,
        pct_start=0.10,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25    # Start at lr/25
    )
    scaler = torch.amp.GradScaler('cuda')

    # Initialize your sophisticated loss function
    hard_loss_fn = DynamicEdgeFocalLoss().to(device)

    # Model saving setup
    model_manager = ModelManager(student_model, device)
    history_logger = HistoryLogger(args)
    wandb_logger = WandbLogger(args)

    best_fbeta = 0.0

    print('\n=== DISTILLATION TRAINING ===')
    print(f'Teacher: {args.teacher_model}')
    print(f'Student: {args.model}')
    print(f'Alpha: {args.alpha}, Temperature: {args.temperature}')
    print(f'Max LR: {args.max_lr}, Epochs: {args.num_epochs}, Batch Size: {args.batch_size}')

    for epoch in range(1, args.num_epochs + 1):
        print(f'\nEPOCH {epoch}')
        print('=' * 80)

        student_model.train()
        teacher_model.eval()  # Keep teacher frozen

        train_tracker = MetricsTracker()
        edge_weight = edge_base_weight * (edge_decay_gamma ** (epoch - 1))

        with tqdm(total=len(train_loader), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, data in enumerate(train_loader):
                data = data.to(device)

                # Apply PointCutMix if enabled (with hard labels)
                pointcutmix_applied = False
                if hasattr(args, 'pointcutmix') and getattr(args, 'pointcutmix', False) and student_model.training:
                    original_data = data

                    from torch_geometric.data import Batch, Data
                    batch_data = Batch.to_data_list(data)
                    batch_pos = [sample.pos for sample in batch_data]
                    batch_reflectance = [sample.reflectance for sample in batch_data]
                    batch_label = [sample.y for sample in batch_data]

                    mixed_pos, mixed_refl, mixed_label = apply_pointcutmix_batch(
                        batch_pos, batch_reflectance, batch_label,
                        prob=getattr(args, 'pointcutmix_prob', 0.25),
                        beta=getattr(args, 'pointcutmix_beta', 1.0),
                        method='spatial'
                    )

                    mixed_batch_data = []
                    for j in range(len(mixed_pos)):
                        mixed_sample = Data(
                            pos=mixed_pos[j],
                            reflectance=mixed_refl[j],
                            y=mixed_label[j],
                            sf=batch_data[j].sf,
                            edge_scores=batch_data[j].edge_scores
                        )
                        mixed_batch_data.append(mixed_sample)

                    data = Batch.from_data_list(mixed_batch_data)
                    pointcutmix_applied = not torch.equal(data.y, original_data.y)

                    if pointcutmix_applied:
                        data.edge_scores = recompute_edge_scores(data.pos, data.y)

                # No automatic label smoothing in distillation - keep labels hard

                try:
                    inputs_ok = True
                    for attr in ("x", "pos", "edge_scores", "y"):
                        if hasattr(data, attr):
                            tensor = getattr(data, attr)
                            if tensor is not None and not torch.isfinite(tensor).all():
                                inputs_ok = False
                                print(f"NaN/Inf detected in {attr}, skipping batch {i}")
                                break

                    if not inputs_ok:
                        continue

                    optimizer.zero_grad()

                    with torch.amp.autocast('cuda'):
                        # Get teacher outputs (no gradients)
                        with torch.no_grad():
                            teacher_outputs = teacher_model(data)

                        # Get student outputs
                        student_outputs = student_model(data)

                        # Compute distillation loss
                        loss = distillation_loss(
                            student_outputs,
                            teacher_outputs.detach(),
                            data.y,
                            data.edge_scores,
                            alpha=args.alpha,
                            temperature=args.temperature,
                            hard_loss_fn=hard_loss_fn
                        )

                    scaler.scale(loss).backward()

                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    loss_value = loss.item()

                    # Check for any unusual outputs that might cause sklearn warnings
                    if torch.isnan(student_outputs).any() or torch.isinf(student_outputs).any():
                        print(f"Warning: NaN/Inf in student outputs at batch {i}")
                        continue

                    train_tracker.update(loss_value, student_outputs, data.y, data.edge_scores)

                    current_metrics = {
                        'loss': train_tracker.loss / max(train_tracker.num_batches, 1),
                        'accuracy': train_tracker.accuracy / max(train_tracker.num_batches, 1),
                        'precision': train_tracker.precision / max(train_tracker.num_batches, 1),
                        'recall': train_tracker.recall / max(train_tracker.num_batches, 1),
                        'f1': train_tracker.f1 / max(train_tracker.num_batches, 1),
                        'miou': train_tracker.miou / max(train_tracker.num_batches, 1)
                    }

                    tepoch.set_postfix({
                        'Lr': optimizer.param_groups[0]["lr"],
                        'Lo': round(current_metrics['loss'], 5),
                        'Scale': f'{scaler.get_scale():.0f}',
                        'Ac': round(current_metrics['accuracy'], 3),
                        'Pr': round(current_metrics['precision'], 3),
                        'Re': round(current_metrics['recall'], 3),
                        'F1': round(current_metrics['f1'], 3),
                        'mIoU': round(current_metrics['miou'], 3),
                        'Alpha': f'{args.alpha:.2f}',
                        'Temp': f'{args.temperature:.1f}'
                    })
                    tepoch.update(1)

                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue

        # Get training metrics
        train_metrics = train_tracker.get_averages()

        # Testing phase - both with and without reflectance (same as train.py)
        if args.test:
            student_model.eval()

            test_tracker_refl = MetricsTracker()
            test_tracker_no_refl = MetricsTracker()

            # Test with reflectance
            with tqdm(total=len(test_loader), colour='cyan', ascii="▒█",
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        data = data.to(device)

                        outputs_refl = student_model(data)
                        test_tracker_refl.update(torch.tensor(0.0), outputs_refl, data.y)

                        curr_metrics_refl = test_tracker_refl.get_averages()
                        tepoch.set_description(f"Val With Refl")
                        tepoch.set_postfix({
                            'Ac': np.around(curr_metrics_refl['accuracy'], 3),
                            'Pr': np.around(curr_metrics_refl['precision'], 3),
                            'Re': np.around(curr_metrics_refl['recall'], 3),
                            'F1': np.around(curr_metrics_refl['f1'], 3),
                            'Fbeta': np.around(curr_metrics_refl['fbeta'], 3),
                            'mIoU': np.around(curr_metrics_refl['miou'], 3),
                        })
                        tepoch.update(1)

            # Test without reflectance
            with tqdm(total=len(test_loader), colour='yellow', ascii="▒█",
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        data = data.to(device)

                        data_no_refl = data.clone()
                        data_no_refl.reflectance = torch.zeros_like(data.reflectance)
                        outputs_no_refl = student_model(data_no_refl)
                        test_tracker_no_refl.update(torch.tensor(0.0), outputs_no_refl, data.y)

                        curr_metrics_no_refl = test_tracker_no_refl.get_averages()
                        tepoch.set_description(f"Val Without Refl")
                        tepoch.set_postfix({
                            'Ac': np.around(curr_metrics_no_refl['accuracy'], 3),
                            'Pr': np.around(curr_metrics_no_refl['precision'], 3),
                            'Re': np.around(curr_metrics_no_refl['recall'], 3),
                            'F1': np.around(curr_metrics_no_refl['f1'], 3),
                            'Fbeta': np.around(curr_metrics_no_refl['fbeta'], 3),
                            'mIoU': np.around(curr_metrics_no_refl['miou'], 3),
                        })
                        tepoch.update(1)

            test_results_refl = test_tracker_refl.get_averages()
            test_results_no_refl = test_tracker_no_refl.get_averages()

            # Harmonic mean of fbeta scores
            harmonic_fbeta = 2 / (1/test_results_refl['fbeta'] + 1/test_results_no_refl['fbeta']) if test_results_refl['fbeta'] > 0 and test_results_no_refl['fbeta'] > 0 else 0

            # Format for wandb compatibility with previous runs
            test_metrics = {
                'fbeta_with_refl': test_results_refl['fbeta'],
                'fbeta_no_refl': test_results_no_refl['fbeta'],
                'harmonic_fbeta': harmonic_fbeta,
                'accuracy': test_results_refl['accuracy'],
                'f1': test_results_refl['f1'],
                'precision': test_results_refl['precision'],
                'recall': test_results_refl['recall']
            }

            # Save best models (same as train.py)
            # Save main model based on with-reflectance fbeta
            if test_results_refl['fbeta'] > best_fbeta:
                best_fbeta = test_results_refl['fbeta']
                save_path = os.path.join(args.wdir, 'model', args.model)
                torch.save({'model_state_dict': student_model.state_dict()}, save_path)
                print(f'Saving best model: {args.model} (Fbeta: {best_fbeta:.4f})')

            # Also save xyz-only model (no-refl version)
            base_name = os.path.splitext(args.model)[0]  # Remove .pth extension
            save_path_xyz = os.path.join(args.wdir, 'model', f'fbeta-xyz-{base_name.replace("fbeta-", "")}.pth')
            torch.save({'model_state_dict': student_model.state_dict()}, save_path_xyz)
        else:
            test_metrics = None

        # Use the same logging as train.py for consistency
        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)

        # Step the scheduler (same as train.py)
        lr_scheduler.step()

    print(f'\nDistillation completed! Best Fbeta: {best_fbeta:.4f}')
    print(f'Student model saved as: {args.model}')