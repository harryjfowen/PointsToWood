import math
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from time import sleep
from torch.optim import AdamW
from src.dataset import create_train_loader, create_test_loader, _fixed_batch_collate
from src.loss import FocalLoss
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.statistics import calculate_harmonic_metrics, print_validation_summary, update_test_metrics_with_harmonic
from src.trainer import EMAModel, run_validation_pass, downsample_batch_to_point_budget, _set_batch_voxel_size, _resolve_amp_config

# Progressive curriculum phase fractions (relative to num_epochs)
_PHASE1_FRAC = 0.22   # Bootstrap: GT-guided, cosine-eased alpha cap
_PHASE2_FRAC = 0.67   # Guided refinement: linear ramp to alpha_final
# Phase 3 (remainder): consolidation with recall-adaptive alpha


class StageProjector(torch.nn.Module):
    """2-layer MLP projector: maps student feature dim → teacher feature dim for feature KD."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid = max(in_dim, out_dim // 2)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, mid, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mid, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def feature_kd_loss(student_model, teacher_model, projectors, stage_indices=(1, 2)):
    """Normalized L2 feature alignment between projected student and teacher encoder stages.

    Uses L2-normalized features so the loss is cosine-distance based (scale-invariant).
    stage_indices: which encoder_stages to align (default: SA2=1, SA3=2).
    """
    student_stages = getattr(student_model, 'encoder_stages', None)
    teacher_stages = getattr(teacher_model, 'encoder_stages', None)
    if not student_stages or not teacher_stages:
        return None

    losses = []
    for proj, sidx in zip(projectors, stage_indices):
        if sidx >= len(student_stages) or sidx >= len(teacher_stages):
            continue
        fs = student_stages[sidx].get('features')
        ft = teacher_stages[sidx].get('features')
        if fs is None or ft is None:
            continue
        n = min(fs.size(0), ft.size(0))
        fs_proj = F.normalize(proj(fs[:n]), dim=1)
        ft_norm = F.normalize(ft[:n].detach(), dim=1)
        losses.append(F.mse_loss(fs_proj, ft_norm))

    if not losses:
        return None
    return torch.stack(losses).mean()


def distillation_loss(student_logits, teacher_logits, targets, edge_scores,
                      alpha=0.7, temperature=3.0, hard_loss_fn=None, hn_factor=1.0,
                      paced=False):
    """Combined distillation loss: soft targets from teacher + hard targets.

    paced=True applies frontier-weighted distillation with GT-agreement gating:
      w = 4*p_s*(1-p_s)  *  [p_t*GT + (1-p_t)*(1-GT)]
    First term (PACED): concentrates KD on student's uncertain frontier (p_s≈0.5).
    Second term (DKD, Zhao et al. CVPR 2022): scales by teacher's confidence in the
    correct GT direction — 1 when teacher is confidently right, 0 when confidently wrong.
    Together: only distil where student is unsure AND teacher is pointing correctly.
    This prevents mode collapse caused by distilling teacher predictions that systematically
    disagree with GT (e.g. pan-EU teacher on domain-shifted biome leaf points).

    Note — T² / alpha interaction: effective soft weight = alpha * T². As T anneals from
    T_start→T_floor the soft contribution shrinks by (T_floor/T_start)² independently of
    alpha, so the alpha ramp-up partially cancels the T² decay. This is intentional
    (T² is the Hinton gradient-scale correction), but be aware the two schedules interact.

    Note — paced vs Phase-3 hard-negative: when paced=True the hard-negative upweighting
    branch (hn_factor) is never reached, so Phase 3 hard-negative correction is silently
    disabled in PACED mode. This is intentional.
    """
    T = max(1.0, float(temperature))
    # Proper binary KD: teacher soft probabilities at temperature T.
    # sigmoid(z/T) softens toward 0.5 as T→∞ (dark knowledge), sharpens as T→1.
    # BCE from temperature-scaled student logits against teacher's soft targets.
    teacher_prob_t = torch.sigmoid(teacher_logits.detach() / T)
    point_loss = F.binary_cross_entropy_with_logits(student_logits / T, teacher_prob_t, reduction='none')

    # Base boundary weights
    if edge_scores is not None and edge_scores.shape == point_loss.shape:
        weights = 1.0 + edge_scores.detach()
    else:
        weights = torch.ones_like(point_loss)

    if paced:
        # Student uncertainty (PACED frontier): peaks at 1 when p=0.5, vanishes when confident.
        p_s = torch.sigmoid(student_logits.detach())
        student_uncertainty = 4.0 * p_s * (1.0 - p_s)

        # GT-agreement weighting (DKD, Zhao et al. CVPR 2022): scale soft loss by how
        # confidently the teacher agrees with the ground-truth label direction.
        #   GT=wood (1): weight = p_teacher  (teacher confidence in wood)
        #   GT=leaf (0): weight = 1-p_teacher (teacher confidence in leaf)
        # = 1 when teacher is perfectly correct, 0.5 when uncertain, 0 when confidently wrong.
        # Prevents the student from learning dark knowledge that points the wrong way —
        # the direct cause of mode collapse onto the dominant class.
        p_t_raw = torch.sigmoid(teacher_logits.detach())
        gt_float = (targets >= 0.5).float()
        gt_agreement = p_t_raw * gt_float + (1.0 - p_t_raw) * (1.0 - gt_float)

        weights = weights * student_uncertainty * gt_agreement
    elif hn_factor > 1.0:
        # Phase 3 hard-negative upweighting (only active when paced=False)
        conf_mask = teacher_logits.detach().abs() > 2.0
        disagree_mask = (student_logits.detach() * teacher_logits.detach()) < 0
        hn_mask = (conf_mask & disagree_mask).float()
        weights = weights * (1.0 + (hn_factor - 1.0) * hn_mask)

    # T² scaling (Hinton et al., 2015): temperature divides gradients by T² relative
    # to hard loss, so multiply soft loss by T² to restore the intended alpha weighting.
    distill_loss = (point_loss * weights).mean() * (T ** 2)

    # Skip hard loss when alpha=1.0 (soft-only path) to avoid wasted computation.
    if hard_loss_fn is not None and alpha < 1.0:
        hard_loss = hard_loss_fn(student_logits, targets)
    else:
        hard_loss = student_logits.new_zeros(())

    total = alpha * distill_loss + (1 - alpha) * hard_loss
    return total, distill_loss, hard_loss


def _curriculum_alpha(epoch: int, phase1_end: int, phase2_end: int, alpha_start: float, alpha_final: float) -> float:
    """Three-phase alpha curriculum to prevent early teacher dominance and late conservative drift.

      Phase 1 (epochs 1–phase1_end): cosine ease-in from 0 → alpha_start.
        Genuinely GT-guided early on; student builds its own representations before
        teacher soft targets carry weight. KD signal grows as student stabilises.
      Phase 2 (phase1_end+1–phase2_end): linear ramp alpha_start → alpha_final.
        Progressive teacher transfer once student representations are stable.
      Phase 3 (phase2_end+1–end): held at alpha_final.
        RecallAdaptiveAlpha may override downward if recall drops.
    """
    if epoch <= phase1_end:
        t = (epoch - 1) / max(1, phase1_end - 1)
        return alpha_start * (1.0 - math.cos(math.pi * t)) / 2.0
    elif epoch <= phase2_end:
        t = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
        return alpha_start + (alpha_final - alpha_start) * t
    else:
        return float(alpha_final)


class RecallAdaptiveAlpha:
    """Phase 3 feedback controller: reduces alpha when recall drops.

    Conservative drift (precision ↑, recall ↓) is caused by high alpha + low T forcing
    the student to match overconfident teacher logits. This controller detects recall
    deficit via EMA and reduces alpha to shift weight back to GT hard loss.
    """
    def __init__(self, alpha_base: float, recall_target: float = 0.85,
                 sensitivity: float = 0.3, min_alpha: float = 0.45):
        self.alpha_base = alpha_base
        self.recall_target = recall_target
        self.sensitivity = sensitivity
        self.min_alpha = min_alpha
        self._recall_ema: float | None = None

    def update(self, recall: float) -> float:
        if self._recall_ema is None:
            self._recall_ema = recall
        else:
            self._recall_ema = 0.7 * self._recall_ema + 0.3 * recall
        deficit = max(0.0, self.recall_target - self._recall_ema)
        return max(self.min_alpha, self.alpha_base - self.sensitivity * deficit)


def _stage_relation_loss(features_s, features_t, batch_idx, num_anchors: int = 64):
    """Channel-agnostic relation KD via cosine affinities to sampled anchors."""
    if features_s is None or features_t is None or batch_idx is None:
        return None

    n = min(features_s.size(0), features_t.size(0), batch_idx.size(0))
    if n < 8:
        return None

    fs = F.normalize(features_s[:n], dim=1)
    ft = F.normalize(features_t[:n], dim=1)
    b = batch_idx[:n]

    loss_terms = []
    for bid in b.unique():
        idx = torch.where(b == bid)[0]
        if idx.numel() < 4:
            continue
        m = min(int(num_anchors), int(idx.numel()))
        anchors = idx[torch.randperm(idx.numel(), device=idx.device)[:m]]
        sim_s = fs[idx] @ fs[anchors].transpose(0, 1)
        sim_t = (ft[idx] @ ft[anchors].transpose(0, 1)).detach()
        loss_terms.append(F.smooth_l1_loss(sim_s, sim_t))

    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def relation_distillation_loss(student_model, teacher_model, num_anchors: int = 64, num_stages: int = 2):
    student_stages = getattr(student_model, 'encoder_stages', None)
    teacher_stages = getattr(teacher_model, 'encoder_stages', None)
    if not student_stages or not teacher_stages:
        return next(student_model.parameters()).new_zeros(())

    n_stages = max(1, min(int(num_stages), len(student_stages), len(teacher_stages)))
    stage_losses = []
    for stage_idx in range(n_stages):
        ss = student_stages[stage_idx]
        ts = teacher_stages[stage_idx]
        stage_loss = _stage_relation_loss(
            ss.get('features', None),
            ts.get('features', None),
            ss.get('batch', None),
            num_anchors=num_anchors,
        )
        if stage_loss is not None:
            stage_losses.append(stage_loss)

    if not stage_losses:
        return next(student_model.parameters()).new_zeros(())
    return torch.stack(stage_losses).mean()


def _collect_stage_gates(model):
    gates = {}
    for short_name, module_name in (("sa1", "sa1_module"), ("sa2", "sa2_module"), ("sa3", "sa3_module")):
        module = getattr(model, module_name, None)
        conv = getattr(module, 'conv', None)
        gate = getattr(conv, 'last_refl_gate_per_point', None)
        if isinstance(gate, torch.Tensor):
            gates[short_name] = gate

    return gates


def _scatter_mean_1d(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Scatter mean for 1-D src: aggregates src values by index."""
    dim_size = int(index.max().item()) + 1
    out = src.new_zeros(dim_size)
    count = src.new_zeros(dim_size)
    out.scatter_add_(0, index, src)
    count.scatter_add_(0, index, torch.ones_like(src))
    return out / count.clamp(min=1.0)


def prototype_kd_loss(student_model, teacher_model, teacher_logits, sa2_projector=None, confidence_threshold: float = 1.5):
    """DINOv2-inspired per-class prototype distillation at SA2.

    Aligns student/teacher class centroids (wood, leaf) using only confidently-predicted
    points (|logit| > threshold). More stable than point-level feature matching on
    ambiguous boundary points — directly addresses complex branch geometry in dense biomes.
    Student features are projected to teacher channel dim before centroid comparison.
    """
    student_stages = getattr(student_model, 'encoder_stages', None)
    teacher_stages = getattr(teacher_model, 'encoder_stages', None)
    if not student_stages or not teacher_stages or len(student_stages) < 2 or len(teacher_stages) < 2:
        return None

    fs = student_stages[1].get('features')
    ft = teacher_stages[1].get('features')
    cluster1_t = teacher_stages[0].get('cluster')  # full_res → SA1
    cluster2_t = teacher_stages[1].get('cluster')  # SA1 → SA2

    if any(x is None for x in [fs, ft, cluster1_t, cluster2_t]):
        return None

    n = min(fs.size(0), ft.size(0))

    # Aggregate full-res teacher logits → SA2 resolution via two-step cluster mapping
    # Voxelsampling is position-based so teacher/student cluster indices are identical
    logits_det = teacher_logits.detach().float()
    n_full = min(logits_det.size(0), cluster1_t.size(0))
    sa1_logits = _scatter_mean_1d(logits_det[:n_full], cluster1_t[:n_full])
    n_sa1 = min(sa1_logits.size(0), cluster2_t.size(0))
    sa2_logits = _scatter_mean_1d(sa1_logits[:n_sa1], cluster2_t[:n_sa1])[:n]

    wood_mask = sa2_logits > confidence_threshold
    leaf_mask = sa2_logits < -confidence_threshold

    losses = []
    for mask in (wood_mask, leaf_mask):
        if mask.sum() < 4:
            continue
        idx = torch.where(mask)[0]
        # Project student prototype to teacher channel dim before comparison
        fs_mean = fs[idx].mean(dim=0, keepdim=True)
        if sa2_projector is not None:
            fs_proto = F.normalize(sa2_projector(fs_mean).squeeze(0), dim=0)
        else:
            fs_proto = F.normalize(fs_mean.squeeze(0), dim=0)
        ft_proto = F.normalize(ft[idx].detach().mean(dim=0), dim=0)
        losses.append(1.0 - (fs_proto * ft_proto).sum())  # cosine distance between centroids

    if not losses:
        return None
    return torch.stack(losses).mean()


def koleo_loss(student_model, num_points: int = 256):
    """KoLeo entropy regularisation on student SA2 features (from DINOv2).

    Maximises minimum pairwise distances between feature representations, preventing
    collapse to a single point in feature space. Applied to a random subset for efficiency.
    """
    student_stages = getattr(student_model, 'encoder_stages', None)
    if not student_stages or len(student_stages) < 2:
        return None

    fs = student_stages[1].get('features')
    if fs is None or fs.size(0) < 8:
        return None

    n = min(fs.size(0), num_points)
    if n < fs.size(0):
        idx = torch.randperm(fs.size(0), device=fs.device)[:n]
        fs_sub = fs[idx]
    else:
        fs_sub = fs

    fs_norm = F.normalize(fs_sub, dim=1)
    sim = fs_norm @ fs_norm.t()
    # Exclude self-similarity by masking diagonal
    sim = sim - 2.0 * torch.eye(n, device=sim.device, dtype=sim.dtype)
    # Minimise maximum cosine similarity → push features apart
    return sim.max(dim=1).values.mean()


def gate_distillation_loss(student_model, teacher_model):
    gates_s = _collect_stage_gates(student_model)
    gates_t = _collect_stage_gates(teacher_model)
    common = [name for name in ("sa1", "sa2", "sa3", "refine") if name in gates_s and name in gates_t]
    if not common:
        return next(student_model.parameters()).new_zeros(())

    losses = []
    for name in common:
        gs = gates_s[name].reshape(-1)
        gt = gates_t[name].reshape(-1).detach()
        n = min(gs.numel(), gt.numel())
        if n < 8:
            continue
        losses.append(F.smooth_l1_loss(gs[:n], gt[:n]))

    if not losses:
        return next(student_model.parameters()).new_zeros(())
    return torch.stack(losses).mean()


def SemanticDistillation(args):
    """Distill a NetFull teacher into a NetLight student, or train NetLight from scratch as a baseline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(False)

    scratch_mode = bool(getattr(args, 'scratch', False))

    # Load teacher using the same config-inference as predicter.py so architecture
    # matches what the checkpoint was trained with.
    from src.predicter import _infer_model_config, load_model as _load_model
    from src.model import NetFull as TeacherNet

    teacher_path = os.path.join(args.wdir, 'model', args.teacher_model)
    teacher_checkpoint = torch.load(teacher_path, map_location=device, weights_only=True)
    teacher_state = teacher_checkpoint['model_state_dict']

    inferred = _infer_model_config(args.teacher_model, teacher_checkpoint, teacher_state)
    print(f'[Teacher] Inferred config: {inferred}')

    teacher_model = TeacherNet(
        num_classes=1,
        C=inferred['c_base'],
        num_kernel_points=inferred['num_kernel_points'],
        learnable_kernels=inferred['learnable_kernels'],
        drop_path_rate=0.0,
        dualnorm_lite=inferred['dualnorm_lite'],
        spatial_mix_lite=inferred['spatial_mix_lite'],
    ).to(device)

    _load_model(teacher_model, teacher_state)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    print(f'Loaded teacher model: {args.teacher_model}')

    # Create student model (drop_path optional for distillation)
    drop_path_rate = getattr(args, 'drop_path_rate', 0.0)
    student_c = int(getattr(args, 'student_c', 16))
    student_kernels = int(getattr(args, 'student_kernels', 16))
    student_learnable_kernels = bool(getattr(args, 'student_learnable_kernels', False))
    dualnorm_lite = bool(getattr(args, 'dualnorm_lite', True))
    spatial_mix_lite = bool(getattr(args, 'spatial_mix_lite', True))
    student_blocks = list(getattr(args, 'student_blocks', [1, 2, 1]))
    sa1_blocks, sa2_blocks, sa3_blocks = int(student_blocks[0]), int(student_blocks[1]), int(student_blocks[2])
    from src.model import NetLight as StudentNet
    student_model = StudentNet(
        num_classes=1,
        C=student_c,
        num_kernel_points=student_kernels,
        learnable_kernels=student_learnable_kernels,
        drop_path_rate=drop_path_rate,
        dualnorm_lite=dualnorm_lite,
        spatial_mix_lite=spatial_mix_lite,
        sa1_blocks=sa1_blocks,
        sa2_blocks=sa2_blocks,
        sa3_blocks=sa3_blocks,
    ).to(device)
    lr = args.max_lr
    weight_decay = args.weight_decay

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f'Teacher: {teacher_params:,} params | Student: {student_params:,} params | Compression: {teacher_params / student_params:.1f}x')

    # Feature KD projectors: student SA2/SA3 → teacher SA2/SA3 channel dims
    feat_kd_weight = float(getattr(args, 'feat_kd_weight', 0.1))
    teacher_C = inferred['c_base']  # inferred from checkpoint — not assumed to be 128
    feat_projectors = torch.nn.ModuleList([
        StageProjector(student_c * 2, teacher_C * 2),  # SA2: student C2 → teacher C2
        StageProjector(student_c * 3, teacher_C * 3),  # SA3: student C3 → teacher C3
    ]).to(device)
    feat_stage_indices = (1, 2)  # encoder_stages indices for SA2, SA3

    train_loader, _ = create_train_loader(args, device)

    if args.test:
        test_loader, test_dataset = create_test_loader(args, device)

    # Log teacher baseline on the biome-specific test set before training starts.
    # This records the teacher→student gap: the key comparison for the specialisation story.
    # Use a sequential loader over ALL voxels (no class balancing) for an unbiased baseline.
    if args.test:
        from torch.utils.data import DataLoader as _DataLoader
        _eval_loader = _DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(8, os.cpu_count() or 4),
            pin_memory=True,
            collate_fn=_fixed_batch_collate,
        )
        print(f'\n{"="*60}')
        print(f'TEACHER BASELINE on {getattr(args, "region", "?")} test set')
        print(f'  ({len(test_dataset)} voxels → {len(_eval_loader)} batches, all data, no class balancing)')
        print(f'{"="*60}')
        teacher_model.eval()
        _t_refl = run_validation_pass(teacher_model, _eval_loader, device, "Teacher With Reflectance", "val_with_reflectance", args)
        _t_norefl = run_validation_pass(teacher_model, _eval_loader, device, "Teacher No Reflectance", "val_no_reflectance", args)
        _t_harm = calculate_harmonic_metrics(_t_refl, _t_norefl)
        print_validation_summary(0, _t_harm)
        print(f'  → Teacher H4-MCC: {_t_harm.get("h4_mcc", float("nan")):.4f} '
              f'(refl={_t_harm.get("mcc_with_refl", float("nan")):.4f}, '
              f'norefl={_t_harm.get("mcc_no_refl", float("nan")):.4f})')
        print(f'{"="*60}\n')
        _teacher_baseline = {
            'h4_mcc': _t_harm.get('h4_mcc', float('nan')),
            'mcc_refl': _t_harm.get('mcc_with_refl', float('nan')),
            'mcc_norefl': _t_harm.get('mcc_no_refl', float('nan')),
            'hauprc': _t_harm.get('harmonic_auprc', float('nan')),
            'auprc_refl': _t_harm.get('auprc_with_refl', float('nan')),
            'auprc_norefl': _t_harm.get('auprc_no_refl', float('nan')),
            'fbeta_refl': _t_harm.get('fbeta_with_refl', float('nan')),
            'fbeta_norefl': _t_harm.get('fbeta_no_refl', float('nan')),
            'brier_refl':   _t_harm.get('brier_with_refl', float('nan')),
            'brier_norefl': _t_harm.get('brier_no_refl',   float('nan')),
        }
        print(f'  → Teacher Brier: {_t_harm.get("mean_brier", float("nan")):.4f} '
              f'(refl={_t_harm.get("brier_with_refl", float("nan")):.4f}, '
              f'norefl={_t_harm.get("brier_no_refl", float("nan")):.4f})')
    else:
        _teacher_baseline = {}

    if getattr(args, 'check_teacher', False):
        print('\n[--check-teacher] Teacher evaluation complete. Exiting before training.')
        return {'teacher_baseline': _teacher_baseline}

    # Hard loss: FocalLoss matching trainer setup.
    # Cyclical gamma helps early training (pure BCE) then focuses hard examples mid-training.
    # label_smoothing=0.1 prevents overconfident outputs — important given teacher soft targets
    # may already push student toward confident predictions.
    focal_alpha = getattr(args, 'alpha_weight', None)  # class weight for wood class; None = no weighting
    label_smoothing = float(getattr(args, 'label_smoothing', 0.1))
    criterion = FocalLoss(
        gamma_max=2.0,
        alpha=focal_alpha,
        label_smoothing=label_smoothing,
        cyclical=True,
        pct_peak=0.33,
        reduction='mean',
    )
    print(f"Loss: FocalLoss cyclical gamma_max=2.0, label_smoothing={label_smoothing}")

    decay_params = []
    no_decay_params = []
    for name, param in student_model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    # Projector params: all are Linear weights (no bias), so always decay
    proj_params = list(feat_projectors.parameters())

    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': proj_params, 'weight_decay': weight_decay},
    ], lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=args.num_epochs,
        pct_start=0.25,  # peak at ~epoch 22 for 90-epoch run — lets student stabilise before teacher ramps
        anneal_strategy='cos',
        div_factor=25
    )

    amp_enabled, amp_dtype, amp_name = _resolve_amp_config(device, getattr(args, 'amp_dtype', 'auto'))
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    print(f"AMP: {amp_name}")

    model_manager = ModelManager(student_model, device)
    history_logger = HistoryLogger(args)
    wandb_logger = WandbLogger(args)

    # EMA is optional (use --ema flag)
    use_ema = getattr(args, 'ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    if use_ema:
        ema_model = EMAModel(student_model, decay=ema_decay)
        ema_model.register()
    else:
        ema_model = None

    # Tracking only — no checkpoints saved for these
    best_auprc_refl, best_auprc_no_refl, best_auprc_harmonic = 0.0, 0.0, 0.0
    best_mcc_refl, best_mcc_no_refl, best_mcc_harmonic = 0.0, 0.0, 0.0
    best_fbeta_refl, best_fbeta_no_refl = 0.0, 0.0
    best_fbeta_edge_refl, best_fbeta_edge_no_refl = 0.0, 0.0
    best_brier_refl, best_brier_no_refl = 1.0, 1.0  # tracking only
    # H4-MCC: primary checkpoint criterion — matches trainer save logic
    best_h4_mcc = 0.0
    alpha_start = float(getattr(args, 'alpha', 0.5))
    alpha_final = float(getattr(args, 'alpha_final', alpha_start))
    relation_kd_weight = float(getattr(args, 'rel_kd_weight', 0.0))
    relation_kd_anchors = int(getattr(args, 'rel_kd_anchors', 64))
    relation_kd_stages = int(getattr(args, 'rel_kd_stages', 2))
    gate_kd_weight = float(getattr(args, 'gate_kd_weight', 0.0))
    proto_kd_weight = float(getattr(args, 'proto_kd_weight', 0.0))
    koleo_weight = float(getattr(args, 'koleo_weight', 0.0))
    T_start = float(getattr(args, 'temperature', 3.0))
    use_paced = bool(getattr(args, 'paced', False))

    # Recall-adaptive alpha controller for Phase 3 drift protection
    recall_controller = RecallAdaptiveAlpha(
        alpha_base=alpha_final,
        recall_target=0.85,
        sensitivity=0.3,
        min_alpha=max(alpha_start, 0.45),
    )
    last_val_recall: float | None = None

    mode_label = 'SCRATCH BASELINE (no KD)' if scratch_mode else 'DISTILLATION TRAINING'
    print(f'\n{"="*60}')
    print(f'{mode_label}  |  region={getattr(args, "region", "?")}')
    print(f'{"="*60}')
    print(f'Teacher: {args.teacher_model} (baseline reference only)' if scratch_mode else f'Teacher: {args.teacher_model}')
    print(f'Student: {args.model}')
    if not scratch_mode:
        paced_str = 'PACED frontier-weighted' if use_paced else 'uniform edge-weighted'
        print(f'Alpha schedule: {alpha_start:.3f} -> {alpha_final:.3f} | Soft loss: BCE with logits, T²-scaled ({paced_str})')
        print(f'Relation KD: w={relation_kd_weight:.3f} | Gate KD: w={gate_kd_weight:.3f} | Feat KD: w={feat_kd_weight:.3f}')
        print(f'Proto KD: w={proto_kd_weight:.3f} | KoLeo: w={koleo_weight:.3f}')
    else:
        print(f'Mode: pure supervised training on biome data — KD losses disabled')
    print(f'Student cfg: C={student_c}, kernels={student_kernels}, learnable_kernels={student_learnable_kernels}, blocks={sa1_blocks}-{sa2_blocks}-{sa3_blocks}, dualnorm_lite={dualnorm_lite}, spatial_mix_lite={spatial_mix_lite}')
    print(f'{"="*60}\n')

    # Normalize checkpoint base name so metric-prefixed model names don't stack prefixes.
    model_dir = os.path.join(args.wdir, 'model')
    raw_model_name = os.path.basename(args.model)
    stem, ext = os.path.splitext(raw_model_name)
    if ext == "":
        ext = ".pth"
    metric_prefixes = (
        "h4mcc-",
        "brier-",
        "mcc-harmonic-", "mcc-xyz-", "mcc-",
        "auprc-harmonic-", "auprc-xyz-", "auprc-",
        "fbeta-edge-xyz-", "fbeta-edge-",
        "fbeta-xyz-", "fbeta-",
    )
    changed = True
    while changed:
        changed = False
        for prefix in metric_prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                changed = True
                break
    ckpt_base = f"{stem}{ext}"

    def _ckpt_path(prefix: str) -> str:
        return os.path.join(model_dir, f"{prefix}{ckpt_base}")
    # Build distill_config once — reused in every checkpoint save
    _distill_config = {
        'teacher_model': args.teacher_model,
        'alpha_start': alpha_start,
        'alpha_final': alpha_final,
        'rel_kd_weight': relation_kd_weight,
        'rel_kd_anchors': relation_kd_anchors,
        'rel_kd_stages': relation_kd_stages,
        'gate_kd_weight': gate_kd_weight,
        'proto_kd_weight': proto_kd_weight,
        'koleo_weight': koleo_weight,
        'student_c': student_c,
        'student_kernels': student_kernels,
        'student_learnable_kernels': student_learnable_kernels,
        'sa1_blocks': sa1_blocks,
        'sa2_blocks': sa2_blocks,
        'sa3_blocks': sa3_blocks,
        'dualnorm_lite': dualnorm_lite,
        'spatial_mix_lite': spatial_mix_lite,
    }

    accumulation_steps = max(1, int(getattr(args, 'accumulation_steps', 4)))
    print(f"Gradient accumulation: {accumulation_steps} steps")
    optimizer.zero_grad(set_to_none=True)

    # Phase boundaries computed once — scale with training length, not hardcoded epochs
    phase1_end = max(5, round(_PHASE1_FRAC * args.num_epochs))
    phase2_end = max(phase1_end + 5, round(_PHASE2_FRAC * args.num_epochs))
    print(f"Alpha curriculum phases: Ph1 epochs 1-{phase1_end} | Ph2 {phase1_end+1}-{phase2_end} | Ph3 {phase2_end+1}-{args.num_epochs}")

    for epoch in range(1, args.num_epochs + 1):
        student_model.train()
        teacher_model.eval()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")

        if ema_model is not None:
            ema_model.set_epoch(epoch)
        criterion.set_epoch(epoch, args.num_epochs)
        train_tracker = MetricsTracker(full_metrics=False)

        # Three-phase curriculum alpha
        alpha_now = _curriculum_alpha(epoch, phase1_end, phase2_end, alpha_start, alpha_final)
        # Phase 3: recall-adaptive override
        if epoch > phase2_end and last_val_recall is not None:
            alpha_now = recall_controller.update(last_val_recall)

        # Temperature anneals from T_start → T_floor over the first 50% of epochs
        T_floor = float(getattr(args, 'temperature_floor', 1.5))
        T_now = max(T_floor, T_start + (T_floor - T_start) * min(1.0, (epoch - 1) / max(1, args.num_epochs * 0.5 - 1)))
        soft_running = 0.0
        hard_running = 0.0
        rel_running = 0.0
        gate_running = 0.0
        feat_running = 0.0
        proto_running = 0.0
        koleo_running = 0.0
        kd_steps = 0

        # Phase 3: halve KoLeo to let representations consolidate without over-regularising
        koleo_w_eff = koleo_weight * 0.5 if epoch > phase2_end else koleo_weight
        # Phase 3: upweight hard negatives (student disagrees with confident teacher) 2.5x
        hn_factor = 2.5 if epoch > phase2_end else 1.0

        max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        epoch_steps = getattr(args, 'epoch_steps', 0)
        total_steps = min(len(train_loader), epoch_steps) if epoch_steps > 0 else len(train_loader)
        accumulated_batches = 0

        with tqdm(total=total_steps, colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, data in enumerate(train_loader):
                if epoch_steps > 0 and i >= epoch_steps:
                    break
                if data.pos.shape[0] > max_points_per_batch:
                    data = downsample_batch_to_point_budget(data, max_points_per_batch, data.pos.device)

                data = data.to(device)
                _set_batch_voxel_size(data, args)

                # Input validation
                inputs_ok = all(
                    torch.isfinite(getattr(data, attr)).all()
                    for attr in ("pos", "edge_scores", "y", "reflectance", "sf")
                    if hasattr(data, attr) and getattr(data, attr) is not None
                )
                if not inputs_ok:
                    print(f"[Warning] Non-finite inputs at step {i}, skipping batch")
                    continue


                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    if not scratch_mode:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(data)
                    else:
                        teacher_outputs = None

                    student_outputs = student_model(data)

                    if not torch.isfinite(student_outputs).all():
                        print(f"[Warning] Non-finite student outputs at step {i}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    if scratch_mode:
                        # Pure supervised: hard loss only, no teacher signal
                        hard_loss = criterion(student_outputs, data.y)
                        soft_loss = student_outputs.new_zeros(())
                        base_loss = hard_loss
                        rel_loss = gate_loss = feat_loss = proto_loss = kl_loss = student_outputs.new_zeros(())
                        loss = hard_loss
                    else:
                        base_loss, soft_loss, hard_loss = distillation_loss(
                            student_outputs, teacher_outputs.detach(),
                            data.y, data.edge_scores,
                            alpha=alpha_now, temperature=T_now,
                            hard_loss_fn=criterion,
                            hn_factor=hn_factor,
                            paced=use_paced,
                        )
                        rel_loss = student_outputs.new_zeros(())
                        if relation_kd_weight > 0.0:
                            rel_loss = relation_distillation_loss(
                                student_model, teacher_model,
                                num_anchors=relation_kd_anchors,
                                num_stages=relation_kd_stages,
                            )

                        gate_loss = student_outputs.new_zeros(())
                        if gate_kd_weight > 0.0:
                            gate_loss = gate_distillation_loss(student_model, teacher_model)

                        feat_loss = student_outputs.new_zeros(())
                        if feat_kd_weight > 0.0:
                            _fl = feature_kd_loss(student_model, teacher_model, feat_projectors, feat_stage_indices)
                            if _fl is not None:
                                feat_loss = _fl

                        proto_loss = student_outputs.new_zeros(())
                        if proto_kd_weight > 0.0:
                            _pl = prototype_kd_loss(student_model, teacher_model, teacher_outputs.detach(),
                                                    sa2_projector=feat_projectors[0])
                            if _pl is not None:
                                proto_loss = _pl

                        kl_loss = student_outputs.new_zeros(())
                        if koleo_w_eff > 0.0:
                            _kl = koleo_loss(student_model)
                            if _kl is not None:
                                kl_loss = _kl

                        loss = (base_loss
                                + relation_kd_weight * rel_loss
                                + gate_kd_weight * gate_loss
                                + feat_kd_weight * feat_loss
                                + proto_kd_weight * proto_loss
                                + koleo_w_eff * kl_loss)
                    loss = torch.clamp(loss, min=0.0, max=10.0)
                    if not torch.isfinite(loss).all():
                        print(f"[Warning] Non-finite distillation loss at step {i}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    loss_to_backward = loss / accumulation_steps
                    scaler.scale(loss_to_backward).backward()
                    accumulated_batches += 1

                    loss_value = loss.item()
                    train_tracker.update(loss_value, student_outputs, data.y, 
                                        edge_scores=data.edge_scores, pos=None)
                    soft_running += float(soft_loss.detach().item())
                    hard_running += float(hard_loss.detach().item())
                    rel_running += float(rel_loss.detach().item())
                    gate_running += float(gate_loss.detach().item())
                    feat_running += float(feat_loss.detach().item())
                    proto_running += float(proto_loss.detach().item())
                    koleo_running += float(kl_loss.detach().item())
                    kd_steps += 1

                    del data, student_outputs, teacher_outputs, loss

                if accumulated_batches >= accumulation_steps:
                    scaler.unscale_(optimizer)
                    _all_params = list(student_model.parameters()) + list(feat_projectors.parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(_all_params, max_norm=1.0)
                    if (not scaler.is_enabled()) and (not torch.isfinite(grad_norm)):
                        print(f"[Warning] Non-finite gradient norm at step {i}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()

                    if ema_model is not None and scaler.get_scale() >= old_scale:
                        ema_model.update()

                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                current_metrics = train_tracker.get_averages()
                tepoch.set_postfix({
                    'Lr': optimizer.param_groups[0]["lr"],
                    'Lo': round(current_metrics['loss'], 5),
                    'KD': round((soft_running / kd_steps) if kd_steps > 0 else 0.0, 4),
                    'BAc': round(current_metrics['accuracy'], 3),
                    'Pr': round(current_metrics['precision'], 3),
                    'Re': round(current_metrics['recall'], 3),
                    'Fb': round(current_metrics['fbeta'], 3),
                })
                tepoch.update(1)
            tepoch.close()

        if accumulated_batches > 0:
            scaler.unscale_(optimizer)
            _all_params = list(student_model.parameters()) + list(feat_projectors.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(_all_params, max_norm=1.0)
            if (not scaler.is_enabled()) and (not torch.isfinite(grad_norm)):
                print("[Warning] Non-finite gradient norm at epoch end, skipping final optimizer step")
                optimizer.zero_grad(set_to_none=True)
            else:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                if ema_model is not None and scaler.get_scale() >= old_scale:
                    ema_model.update()

                optimizer.zero_grad(set_to_none=True)

        lr_scheduler.step()
        train_metrics = train_tracker.get_averages()
        if kd_steps > 0:
            if scratch_mode:
                print(f"Scratch: hard={hard_running / kd_steps:.4f}")
            else:
                phase = 1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3)
                paced_tag = ' PACED' if use_paced else ''
                _avg = lambda x: x / kd_steps
                _aux = []
                if _avg(rel_running)   > 1e-4: _aux.append(f"rel={_avg(rel_running):.4f}")
                if _avg(gate_running)  > 1e-4: _aux.append(f"gate={_avg(gate_running):.4f}")
                if _avg(feat_running)  > 1e-4: _aux.append(f"feat={_avg(feat_running):.4f}")
                if _avg(proto_running) > 1e-4: _aux.append(f"proto={_avg(proto_running):.4f}")
                if _avg(koleo_running) > 1e-4: _aux.append(f"koleo={_avg(koleo_running):.4f}")
                _aux_str = " | " + " | ".join(_aux) if _aux else ""
                print(
                    f"KD [Ph{phase}{paced_tag}] alpha={alpha_now:.3f} T={T_now:.2f} | "
                    f"soft={_avg(soft_running):.4f} hard={_avg(hard_running):.4f}"
                    f"{_aux_str}"
                )

        # Kernel diagnostics
        kernel_metrics = []
        for module in student_model.modules():
            if hasattr(module, 'kernel_entropy') and hasattr(module, 'active_kernels'):
                kernel_metrics.append((module.kernel_entropy.item(), module.active_kernels.item()))
        if kernel_metrics:
            avg_entropy = sum(m[0] for m in kernel_metrics) / len(kernel_metrics)
            avg_active = sum(m[1] for m in kernel_metrics) / len(kernel_metrics)
            max_active = max(m[1] for m in kernel_metrics)
            print(f"Kernels: {avg_active:.1f} avg, {max_active:.1f} max | Entropy: {avg_entropy:.2f}")

        if args.test:
            student_model.eval()
            if ema_model is not None:
                ema_model.apply_shadow()

            sleep(0.1)
            test_metrics_with_refl = run_validation_pass(student_model, test_loader, device, "With Reflectance", "val_with_reflectance", args)
            sleep(0.1)
            test_metrics_no_refl = run_validation_pass(student_model, test_loader, device, "No Reflectance", "val_no_reflectance", args)

            harmonic_metrics = calculate_harmonic_metrics(test_metrics_with_refl, test_metrics_no_refl)
            print_validation_summary(epoch, harmonic_metrics)

            test_metrics = update_test_metrics_with_harmonic(test_metrics_with_refl.copy(), harmonic_metrics)

            # Track recall for Phase 3 RecallAdaptiveAlpha
            last_val_recall = test_metrics_with_refl.get('recall', None)

        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)

        # Single checkpoint: best H4-MCC — matches trainer save criterion exactly.
        # H4-MCC is the 4-way harmonic mean across (with/no refl) × (pure/edge).
        # All other metrics tracked for reporting only — no separate checkpoints.
        if args.test and epoch > int(args.num_epochs * 0.25):
            # Track-only (reporting)
            best_auprc_refl    = max(best_auprc_refl,    harmonic_metrics['auprc_with_refl'])
            best_auprc_no_refl = max(best_auprc_no_refl, harmonic_metrics['auprc_no_refl'])
            best_auprc_harmonic= max(best_auprc_harmonic, harmonic_metrics['harmonic_auprc'])
            best_mcc_refl      = max(best_mcc_refl,      harmonic_metrics['mcc_with_refl'])
            best_mcc_no_refl   = max(best_mcc_no_refl,   harmonic_metrics['mcc_no_refl'])
            best_mcc_harmonic  = max(best_mcc_harmonic,  harmonic_metrics['harmonic_mcc'])
            best_fbeta_refl    = max(best_fbeta_refl,    harmonic_metrics['fbeta_with_refl'])
            best_fbeta_no_refl = max(best_fbeta_no_refl, harmonic_metrics['fbeta_no_refl'])
            best_fbeta_edge_refl    = max(best_fbeta_edge_refl,    harmonic_metrics['fbeta_edge_with_refl'])
            best_fbeta_edge_no_refl = max(best_fbeta_edge_no_refl, harmonic_metrics['fbeta_edge_no_refl'])
            if harmonic_metrics['brier_with_refl'] < best_brier_refl:
                best_brier_refl = harmonic_metrics['brier_with_refl']
            if harmonic_metrics['brier_no_refl'] < best_brier_no_refl:
                best_brier_no_refl = harmonic_metrics['brier_no_refl']
            # Checkpoint: H4-MCC (4-way harmonic — penalises worst-case condition)
            h4 = harmonic_metrics.get('h4_mcc', 0.0)
            if h4 > best_h4_mcc:
                best_h4_mcc = h4
                ckpt_path = _ckpt_path('h4mcc-')
                torch.save(
                    {
                        'model_state_dict': (ema_model.shadow if ema_model is not None else student_model).state_dict(),
                        'distill_config': _distill_config,
                    },
                    ckpt_path,
                )
                print(f'Saved best student (H4-MCC={best_h4_mcc:.4f}): {ckpt_path}')

        # Restore EMA after all saves so training resumes with live weights
        if ema_model is not None and args.test:
            ema_model.restore()

        if epoch == args.num_epochs:
            if ema_model is not None:
                ema_model.apply_shadow()
            print("Saving final student model")
            torch.save(
                {
                    'model_state_dict': student_model.state_dict(),
                    'distill_config': _distill_config,
                },
                os.path.join(args.wdir, 'model', args.model),
            )

    print(f'\nDistillation completed!')

    return {
        'best_mcc_harmonic': best_mcc_harmonic,
        'best_mcc_refl': best_mcc_refl,
        'best_mcc_no_refl': best_mcc_no_refl,
        'best_auprc_harmonic': best_auprc_harmonic,
        'best_auprc_refl': best_auprc_refl,
        'best_auprc_no_refl': best_auprc_no_refl,
        'best_fbeta_refl': best_fbeta_refl,
        'best_fbeta_no_refl': best_fbeta_no_refl,
        'best_fbeta_edge_refl': best_fbeta_edge_refl,
        'best_fbeta_edge_no_refl': best_fbeta_edge_no_refl,
        'best_h4_mcc': best_h4_mcc,
        'best_brier_refl': best_brier_refl,
        'best_brier_no_refl': best_brier_no_refl,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': teacher_params / student_params,
        'scratch_mode': scratch_mode,
        'teacher_baseline': _teacher_baseline,
    }
