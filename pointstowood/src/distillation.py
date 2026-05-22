import math
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from time import sleep
from torch.optim import AdamW
from src.dataset import create_train_loader, create_test_loader, _fixed_batch_collate
from src.loss import FocalLoss, ReflectanceFPPenalty, SupConLoss
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.statistics import calculate_harmonic_metrics, print_validation_summary, update_test_metrics_with_harmonic
from src.trainer import (
    EMAModel,
    VoxelDifficultyTracker,
    _apply_train_sampler_weights,
    _per_sample_boundary_weighted_bce,
    run_validation_pass,
    run_eval_visualization,
    downsample_batch_to_point_budget,
    _set_batch_voxel_size,
    _resolve_amp_config,
)

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
        try:
            hard_loss = hard_loss_fn(student_logits, targets, edge_scores=edge_scores)
        except TypeError:
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


def _fmt_kernel_points(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().view(-1).tolist()
    if isinstance(value, (list, tuple)):
        return "/".join(str(int(v)) for v in value)
    return str(int(value))


def _fmt_params(value):
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _fmt_steps(value):
    try:
        value = int(value)
    except Exception:
        return str(value)
    return "all" if value <= 0 else str(value)


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

    teacher_model = TeacherNet(
        num_classes=1,
        C=inferred['c_base'],
        num_kernel_points=inferred['stage_kernel_points'],
        learnable_kernels=inferred['learnable_kernels'],
        drop_path_rate=0.0,
        spatial_mix_lite=inferred['spatial_mix_lite'],
        flash_dim=inferred.get('flash_dim', 32),
        memory_efficient_conv=inferred.get('memory_efficient_conv', False),
        sparse_max=inferred.get('sparse_max', True),
        compressed_head=inferred['compressed_head'],
        compressed_head_dim=inferred['compressed_head_dim'],
        k_neighbors=inferred['k_neighbors'],
    ).to(device)

    _load_model(teacher_model, teacher_state)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # Create student model (drop_path optional for distillation)
    drop_path_rate = getattr(args, 'drop_path_rate', 0.0)
    student_c = int(getattr(args, 'student_c', 32))
    _sk = getattr(args, 'student_kernels', [16, 16, 16])
    student_kernels = tuple(_sk) if isinstance(_sk, (list, tuple)) else int(_sk)
    student_k_neighbors = int(getattr(args, 'student_k_neighbors', 32))
    student_learnable_kernels = bool(getattr(args, 'student_learnable_kernels', False))
    spatial_mix_lite = bool(getattr(args, 'spatial_mix_lite', True))
    student_blocks = list(getattr(args, 'student_blocks', [2, 3, 1]))
    sa1_blocks, sa2_blocks, sa3_blocks = int(student_blocks[0]), int(student_blocks[1]), int(student_blocks[2])
    student_compressed_head = bool(getattr(args, 'student_compressed_head', True))
    student_head_dim = int(getattr(args, 'student_head_dim', 64))
    from src.model import NetLight as StudentNet
    student_model = StudentNet(
        num_classes=1,
        C=student_c,
        num_kernel_points=student_kernels,
        learnable_kernels=student_learnable_kernels,
        drop_path_rate=drop_path_rate,
        spatial_mix_lite=spatial_mix_lite,
        sa1_blocks=sa1_blocks,
        sa2_blocks=sa2_blocks,
        sa3_blocks=sa3_blocks,
        k_neighbors=student_k_neighbors,
        compressed_head=student_compressed_head,
        compressed_head_dim=student_head_dim,
    ).to(device)
    lr = args.max_lr
    weight_decay = args.weight_decay

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    # Feature KD projectors: student SA2/SA3 → teacher SA2/SA3 channel dims
    feat_kd_weight = float(getattr(args, 'feat_kd_weight', 0.0))
    teacher_C = inferred['c_base']  # inferred from checkpoint — not assumed to be 128
    feat_projectors = torch.nn.ModuleList([
        StageProjector(student_c * 2, teacher_C * 2),  # SA2: student C2 → teacher C2
        StageProjector(student_c * 3, teacher_C * 3),  # SA3: student C3 → teacher C3
    ]).to(device)
    feat_stage_indices = (1, 2)  # encoder_stages indices for SA2, SA3

    loader_verbose = getattr(args, 'verbose', False)
    args.verbose = False
    try:
        train_loader, train_dataset = create_train_loader(args, device)
        if args.test:
            test_loader, test_dataset = create_test_loader(args, device)
        else:
            test_loader, test_dataset = None, None
    finally:
        args.verbose = loader_verbose

    _teacher_baseline = {}

    # Hard loss: same focal/boundary machinery as the main trainer, with KD-specific
    # soft loss layered on top.
    gamma_max = float(getattr(args, 'gamma_max', 4.0))
    gamma_peak_pct = min(0.95, max(0.05, float(getattr(args, 'gamma_peak_pct', 0.33))))
    focal_alpha = getattr(args, 'focal_alpha', getattr(args, 'alpha_weight', None))
    label_smoothing = float(getattr(args, 'label_smoothing', 0.05))
    boundary_max = float(getattr(args, 'boundary_weight', 0.0))
    boundary_ramp_start = float(getattr(args, 'boundary_ramp_start', 0.1))
    criterion = FocalLoss(
        gamma_max=gamma_max,
        alpha=focal_alpha,
        label_smoothing=label_smoothing,
        cyclical=True,
        pct_peak=gamma_peak_pct,
        boundary_max=boundary_max,
        boundary_ramp_start=boundary_ramp_start,
        reduction='mean',
    )
    refl_fp_weight = float(getattr(args, 'refl_fp_penalty', 0.05))

    refl_fp_criterion = (
        ReflectanceFPPenalty(
            margin=1.0,
            strength=2.0,
            weight=refl_fp_weight,
            ramp=bool(getattr(args, 'refl_fp_ramp', True)),
            flat=bool(getattr(args, 'refl_fp_flat', False)),
        )
        if refl_fp_weight > 0
        else None
    )

    contrastive_weight = float(getattr(args, 'contrastive_weight', 0.05))
    contrastive_criterion = (
        SupConLoss(n_anchors=512, start_temp=0.2, end_temp=0.07, weight=contrastive_weight, ramp_frac=0.15)
        if contrastive_weight > 0
        else None
    )

    per_voxel_on = bool(getattr(args, 'per_voxel_difficulty', getattr(args, 'difficulty_mining', False)))
    per_voxel_alpha = float(getattr(args, 'per_voxel_alpha', 2.0))
    per_voxel_warmup = int(getattr(args, 'per_voxel_warmup_epochs', max(3, int(round(0.10 * args.num_epochs)))))
    voxel_ema_alpha = float(getattr(args, 'voxel_ema_alpha', 0.9))
    per_voxel_summary = "off"
    if per_voxel_on:
        sampler = getattr(train_loader, 'batch_sampler', None)
        if sampler is None or not hasattr(sampler, 'set_weights'):
            per_voxel_on = False
            per_voxel_summary = "requested, disabled (sampler has no set_weights)"
        else:
            prior = np.asarray(
                getattr(train_dataset, 'edge_fractions', np.zeros(len(train_dataset.keys))),
                dtype=np.float64,
            )
            voxel_tracker = VoxelDifficultyTracker(
                n_real=len(train_dataset.keys),
                prior=prior,
                alpha_ema=voxel_ema_alpha,
            )
            per_voxel_summary = (
                f"on (alpha={per_voxel_alpha:g}, warmup={per_voxel_warmup}, "
                f"ema={voxel_ema_alpha:g})"
            )
    if not per_voxel_on:
        voxel_tracker = None

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

    # Epoch-level OneCycleLR: total_steps is epochs, and step() is called once
    # after each epoch rather than inside the batch loop.
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
    relation_kd_stages = int(getattr(args, 'rel_kd_stages', 3))
    gate_kd_weight = float(getattr(args, 'gate_kd_weight', 0.0))
    proto_kd_weight = float(getattr(args, 'proto_kd_weight', 0.0))
    koleo_weight = float(getattr(args, 'koleo_weight', 0.0))
    T_start = float(getattr(args, 'temperature', 3.0))
    use_paced = bool(getattr(args, 'paced', False))
    teacher_model.capture_encoder_features = (
        relation_kd_weight > 0.0 or feat_kd_weight > 0.0 or proto_kd_weight > 0.0
    )

    # Recall-adaptive alpha controller for Phase 3 drift protection
    recall_controller = RecallAdaptiveAlpha(
        alpha_base=alpha_final,
        recall_target=0.85,
        sensitivity=0.3,
        min_alpha=max(alpha_start, 0.45),
    )
    last_val_recall: float | None = None

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
        'gamma_max': gamma_max,
        'gamma_peak_pct': gamma_peak_pct,
        'label_smoothing': label_smoothing,
        'boundary_weight': boundary_max,
        'refl_fp_penalty': refl_fp_weight,
        'contrastive_weight': contrastive_weight,
        'density_aug': bool(getattr(args, 'density_aug', False)),
        'density_aug_prob': float(getattr(args, 'density_aug_prob', 0.20)),
        'per_voxel_difficulty': per_voxel_on,
        'student_c': student_c,
        'student_kernels': student_kernels,
        'student_k_neighbors': student_k_neighbors,
        'student_learnable_kernels': student_learnable_kernels,
        'sa1_blocks': sa1_blocks,
        'sa2_blocks': sa2_blocks,
        'sa3_blocks': sa3_blocks,
        'student_compressed_head': student_compressed_head,
        'student_head_dim': student_head_dim,
        'spatial_mix_lite': spatial_mix_lite,
    }
    _model_config = {
        'model_family': getattr(student_model, 'model_family', 'light'),
        'c_base': getattr(student_model, 'c_base', student_c),
        'k_neighbors': getattr(student_model, 'k_neighbors', student_k_neighbors),
        'stage_kernel_points': list(getattr(student_model, 'stage_kernel_points', student_kernels)),
        'num_kernel_points': getattr(student_model, 'num_kernel_points', student_kernels[0] if isinstance(student_kernels, tuple) else student_kernels),
        'learnable_kernels': getattr(student_model, 'learnable_kernels', student_learnable_kernels),
        'spatial_mix_lite': getattr(student_model, 'spatial_mix_lite', spatial_mix_lite),
        'flash_dim': getattr(student_model, 'flash_dim', 32),
        'memory_efficient_conv': getattr(student_model, 'memory_efficient_conv', False),
        'sparse_max': getattr(student_model, 'sparse_max', True),
        'sa1_blocks': getattr(student_model, 'sa1_blocks', sa1_blocks),
        'sa2_blocks': getattr(student_model, 'sa2_blocks', sa2_blocks),
        'sa3_blocks': getattr(student_model, 'sa3_blocks', sa3_blocks),
        'compressed_head': getattr(student_model, 'compressed_head', student_compressed_head),
        'compressed_head_dim': getattr(student_model, 'compressed_head_dim', student_head_dim),
    }

    accumulation_steps = max(1, int(getattr(args, 'accumulation_steps', 4)))
    ema_str = f" | EMA(decay={ema_decay})" if use_ema else ""
    optimizer.zero_grad(set_to_none=True)
    global_train_step = 0
    wandb_train_log_interval = max(1, int(getattr(args, 'wandb_train_log_interval', 25)))

    # Phase boundaries computed once — scale with training length, not hardcoded epochs
    fast_distil = bool(getattr(args, 'fast_distil', False))
    if fast_distil:
        # Skip warm-up entirely: 1 epoch ease-in, 2 epoch ramp, rest at full alpha.
        # Designed for 10-epoch domain-specific runs where the teacher is already strong.
        phase1_end = 1
        phase2_end = min(3, args.num_epochs)
        args.eval_interval = 1
    else:
        phase1_end = max(5, round(_PHASE1_FRAC * args.num_epochs))
        phase2_end = max(phase1_end + 5, round(_PHASE2_FRAC * args.num_epochs))
    T_floor = float(getattr(args, 'temperature_floor', 1.5))

    mode_label = "scratch" if scratch_mode else ("distill+PACED" if use_paced else "distill")
    teacher_head = f"compressed:{inferred['compressed_head_dim']}" if inferred['compressed_head'] else "full"
    student_head = f"compressed:{student_head_dim}" if student_compressed_head else "full"
    region_prefixes = getattr(args, 'region_prefixes', '?')
    train_file_count = getattr(args, 'train_file_count', '?')
    test_file_count = getattr(args, 'test_file_count', '?')
    train_voxels = len(getattr(train_dataset, 'keys', train_dataset))
    test_voxels = len(getattr(test_dataset, 'keys', test_dataset)) if test_dataset is not None else 0
    train_batches = len(train_loader)
    test_batches = len(test_loader) if test_loader is not None else 0
    grid_size = getattr(args, 'grid_size', ['?'])
    grid_str = "/".join(f"{float(v):g}m" for v in grid_size) if isinstance(grid_size, (list, tuple)) else f"{float(grid_size):g}m"
    soft_mode = "none" if scratch_mode else ("PACED frontier" if use_paced else "uniform")
    density_aug = "off"
    if bool(getattr(args, 'density_aug', False)):
        density_aug = f"on (p={float(getattr(args, 'density_aug_prob', 0.0)):g})"
    eval_summary = "off"
    if bool(getattr(args, 'eval', False)):
        any_wood = getattr(args, 'eval_any_wood', None)
        any_wood_str = "argmax" if any_wood is None else f"any_wood={float(any_wood):g}"
        eval_summary = (
            f"{getattr(args, 'eval_file_count', '?')} file(s) | "
            f"model_grid={float(getattr(args, 'eval_grid_size', 2.0)):g}m | "
            f"collect={float(getattr(args, 'eval_collect_grid_size', 0.04)):g}m | "
            f"{any_wood_str} | interval={int(getattr(args, 'eval_interval', 10))}"
        )
    aux_bits = [
        f"refl_fp={refl_fp_weight:g}" if refl_fp_weight > 0 else "refl_fp=off",
        f"supcon={contrastive_weight:g}" if contrastive_weight > 0 else "supcon=off",
    ]
    kd_bits = []
    if relation_kd_weight > 0: kd_bits.append(f"rel={relation_kd_weight:g}")
    if gate_kd_weight > 0: kd_bits.append(f"gate={gate_kd_weight:g}")
    if feat_kd_weight > 0: kd_bits.append(f"feat={feat_kd_weight:g}")
    if proto_kd_weight > 0: kd_bits.append(f"proto={proto_kd_weight:g}")
    if koleo_weight > 0: kd_bits.append(f"koleo={koleo_weight:g}")
    kd_extra = ", ".join(kd_bits) if kd_bits else "extra_kd=off"
    hard_desc = f"focal(gamma={gamma_max:g}, smooth={label_smoothing:g})"
    if boundary_max > 0:
        hard_desc += f", boundary={boundary_max:g}"

    print("\n" + "=" * 72)
    print("DISTILLATION SETUP")
    print("=" * 72)
    print(f"Mode      : {mode_label} | device={device} | amp={amp_name}{ema_str}")
    print(f"Region    : {getattr(args, 'region', '?')} (prefixes: {region_prefixes}) | files train/test={train_file_count}/{test_file_count}")
    print(f"Data      : train {train_voxels:,} voxels -> {train_batches:,} batches | test {test_voxels:,} voxels -> {test_batches:,} batches")
    print(f"Voxel     : grid={grid_str} | min_pts={getattr(args, 'min_pts', '?')} | max_pts={getattr(args, 'max_pts', '?')} | batch_pts={getattr(args, 'max_points_per_batch', '?')} | packing={getattr(args, 'packing_mode', '?')}")
    print(f"Teacher   : {args.teacher_model} | NetFull C={inferred['c_base']} K={_fmt_kernel_points(inferred['stage_kernel_points'])} kNN={inferred['k_neighbors']} blocks={inferred['sa1_blocks']}-{inferred['sa2_blocks']}-{inferred['sa3_blocks']} head={teacher_head} | {_fmt_params(teacher_params)} params")
    print(f"Student   : {args.model} | NetLight C={student_c} K={_fmt_kernel_points(student_kernels)} kNN={student_k_neighbors} blocks={sa1_blocks}-{sa2_blocks}-{sa3_blocks} head={student_head} | {_fmt_params(student_params)} params ({teacher_params / student_params:.1f}x)")
    print(f"Schedule  : epochs={args.num_epochs} | steps/epoch={_fmt_steps(getattr(args, 'epoch_steps', 0))} | val_steps={_fmt_steps(getattr(args, 'val_steps', 0))} | accum={accumulation_steps} | lr={lr:g} | wd={weight_decay:g}")
    if scratch_mode:
        print(f"Loss      : supervised {hard_desc} | {', '.join(aux_bits)}")
    else:
        print(f"Loss      : soft={soft_mode} alpha={alpha_start:g}->{alpha_final:g} T={T_start:g}->{T_floor:g} | hard={hard_desc}")
        print(f"Aux/KD    : {', '.join(aux_bits)} | {kd_extra}")
    print(f"Sampling  : balance={getattr(args, 'balance_mode', '?')} | hard_mining={per_voxel_summary} | density_aug={density_aug}")
    print(f"Eval      : {eval_summary}")
    print(f"Phases    : 1-{phase1_end} | {phase1_end + 1}-{phase2_end} | {phase2_end + 1}-{args.num_epochs}")
    print("=" * 72)

    # Log teacher baseline on the biome-specific test set before training starts.
    # Use a sequential loader over all voxels so the teacher reference is not class-balanced.
    if args.test:
        from torch.utils.data import DataLoader as _DataLoader
        _eval_loader = _DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=_fixed_batch_collate,
        )
        print(
            f"\nTeacher baseline: {getattr(args, 'region', '?')} test | "
            f"{len(test_dataset):,} voxels -> {len(_eval_loader):,} batches | "
            f"val_steps={_fmt_steps(getattr(args, 'val_steps', 0))} | reflectance only"
        )
        teacher_model.eval()
        _t_refl = run_validation_pass(teacher_model, _eval_loader, device, "Teacher With Reflectance", "val_with_reflectance", args)
        _t_norefl = _t_refl.copy()
        _t_harm = calculate_harmonic_metrics(_t_refl, _t_norefl)
        print(
            "Teacher summary: "
            f"H4-MCC={_t_harm.get('h4_mcc', float('nan')):.4f} | "
            f"MCC={_t_harm.get('mcc_with_refl', float('nan')):.4f} | "
            f"Brier={_t_harm.get('mean_brier', float('nan')):.4f}"
        )
        _teacher_baseline = {
            'h4_mcc': _t_harm.get('h4_mcc', float('nan')),
            'mcc_refl': _t_harm.get('mcc_with_refl', float('nan')),
            'mcc_norefl': _t_harm.get('mcc_no_refl', float('nan')),
            'hauprc': _t_harm.get('harmonic_auprc', float('nan')),
            'auprc_refl': _t_harm.get('auprc_with_refl', float('nan')),
            'auprc_norefl': _t_harm.get('auprc_no_refl', float('nan')),
            'fbeta_refl': _t_harm.get('fbeta_with_refl', float('nan')),
            'fbeta_norefl': _t_harm.get('fbeta_no_refl', float('nan')),
            'brier_refl': _t_harm.get('brier_with_refl', float('nan')),
            'brier_norefl': _t_harm.get('brier_no_refl', float('nan')),
        }

    if bool(getattr(args, 'eval', False)):
        print(
            f"\nTeacher eval-folder baseline: grid={float(getattr(args, 'eval_grid_size', 2.0)):g}m | "
            f"collect={float(getattr(args, 'eval_collect_grid_size', 0.04)):g}m | "
            f"any_wood={getattr(args, 'eval_any_wood', None)}"
        )
        run_eval_visualization(teacher_model, args, device, 0, save_outputs=False)

    if getattr(args, 'check_teacher', False):
        print("[check-teacher] Teacher evaluation complete. Exiting before training.")
        return {'teacher_baseline': _teacher_baseline}

    for epoch in range(1, args.num_epochs + 1):
        student_model.train()
        teacher_model.eval()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")

        if ema_model is not None and hasattr(ema_model, 'set_epoch'):
            ema_model.set_epoch(epoch)
        criterion.set_epoch(epoch, args.num_epochs)
        if refl_fp_criterion is not None:
            refl_fp_criterion.set_epoch(epoch, args.num_epochs)
        if contrastive_criterion is not None:
            contrastive_criterion.set_epoch(epoch, args.num_epochs)

        if per_voxel_on and voxel_tracker is not None:
            voxel_alpha_cur = per_voxel_alpha if epoch > per_voxel_warmup else 0.0
            coverage_warmup = epoch <= per_voxel_warmup
            unseen_boost = float(getattr(args, 'unseen_voxel_boost', 8.0)) if coverage_warmup else 0.0
            allow_replacement = not (
                coverage_warmup and not bool(getattr(args, 'coverage_warmup_replacement', False))
            )
            _apply_train_sampler_weights(
                train_loader,
                train_dataset,
                voxel_difficulty=voxel_tracker if voxel_alpha_cur > 0 else None,
                voxel_alpha=voxel_alpha_cur,
                coverage_tracker=voxel_tracker if coverage_warmup else None,
                unseen_boost=unseen_boost,
                allow_replacement=allow_replacement,
            )
            if epoch > per_voxel_warmup:
                diff_raw = voxel_tracker.normalised_weights()
                train_dataset.difficulty_scores = np.clip(diff_raw - 1.0, 0.0, 1.0)

        train_tracker = MetricsTracker(full_metrics=False)

        # Three-phase curriculum alpha
        alpha_now = _curriculum_alpha(epoch, phase1_end, phase2_end, alpha_start, alpha_final)
        # Phase 3: recall-adaptive override
        if epoch > phase2_end and last_val_recall is not None:
            alpha_now = recall_controller.update(last_val_recall)

        # Temperature anneals from T_start to T_floor over the first 50% of epochs
        T_now = max(T_floor, T_start + (T_floor - T_start) * min(1.0, (epoch - 1) / max(1, args.num_epochs * 0.5 - 1)))
        soft_running = 0.0
        hard_running = 0.0
        rel_running = 0.0
        gate_running = 0.0
        feat_running = 0.0
        proto_running = 0.0
        koleo_running = 0.0
        refl_fp_running = 0.0
        con_running = 0.0
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

                if getattr(args, 'density_aug', False):
                    from src.augmentation import random_density_downsample_batch
                    spacing = getattr(args, 'density_aug_spacing', [0.01, 0.04])
                    random_density_downsample_batch(
                        data,
                        spacing_min=spacing[0],
                        spacing_max=spacing[1],
                        prob=getattr(args, 'density_aug_prob', 0.20),
                        difficulty=getattr(data, 'difficulty', None),
                        hard_skip_threshold=getattr(args, 'density_aug_hard_threshold', 0.75),
                        difficulty_power=getattr(args, 'density_aug_difficulty_power', 2.0),
                    )

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
                        hard_loss = criterion(student_outputs, data.y, edge_scores=getattr(data, 'edge_scores', None))
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

                    refl_fp_loss = student_outputs.new_zeros(())
                    if refl_fp_criterion is not None and getattr(data, 'reflectance', None) is not None:
                        refl_fp_loss = refl_fp_criterion(student_outputs, data.y.float(), data.reflectance)
                        loss = loss + refl_fp_loss

                    con_loss = student_outputs.new_zeros(())
                    if contrastive_criterion is not None and hasattr(student_model, 'last_proj'):
                        _con = contrastive_criterion(
                            student_model.last_proj,
                            (data.y.float() >= 0.5),
                            edge_scores=getattr(data, 'edge_scores', None),
                        )
                        if torch.isfinite(_con):
                            con_loss = _con
                            loss = loss + con_loss

                    if per_voxel_on and voxel_tracker is not None:
                        with torch.no_grad():
                            per_sample_loss = _per_sample_boundary_weighted_bce(
                                student_outputs.detach(),
                                data.y.float(),
                                data.batch,
                                edge_scores=getattr(data, 'edge_scores', None),
                            )
                            voxel_ids = getattr(data, 'voxel_idx', None)
                            if voxel_ids is not None:
                                voxel_ids_np = voxel_ids.view(-1).detach().cpu().numpy()
                                voxel_tracker.update(
                                    voxel_ids_np[:per_sample_loss.numel()],
                                    per_sample_loss.detach().cpu().numpy(),
                                )

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
                    refl_fp_running += float(refl_fp_loss.detach().item())
                    con_running += float(con_loss.detach().item())
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

                global_train_step += 1
                if (
                    wandb_logger.wandb is not None
                    and (global_train_step % wandb_train_log_interval == 0 or (i + 1) == total_steps)
                ):
                    wandb_logger.log_train_step(
                        global_train_step,
                        optimizer.param_groups[0]["lr"],
                        current_metrics,
                        batch_loss=loss_value,
                        extra_metrics={
                            "train/kd_loss": (soft_running / kd_steps) if kd_steps > 0 else 0.0,
                        },
                    )
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

        # Keep LR cycling per epoch, not within the epoch.
        lr_scheduler.step()
        train_metrics = train_tracker.get_averages()
        if kd_steps > 0:
            if scratch_mode:
                _avg = lambda x: x / kd_steps
                _aux = []
                if _avg(refl_fp_running) > 1e-4: _aux.append(f"refl_fp={_avg(refl_fp_running):.4f}")
                if _avg(con_running) > 1e-4: _aux.append(f"supcon={_avg(con_running):.4f}")
                _aux_str = " | " + " | ".join(_aux) if _aux else ""
                print(f"Scratch: hard={_avg(hard_running):.4f}{_aux_str}")
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
                if _avg(refl_fp_running) > 1e-4: _aux.append(f"refl_fp={_avg(refl_fp_running):.4f}")
                if _avg(con_running) > 1e-4: _aux.append(f"supcon={_avg(con_running):.4f}")
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
            test_metrics_no_refl = test_metrics_with_refl.copy()

            harmonic_metrics = calculate_harmonic_metrics(test_metrics_with_refl, test_metrics_no_refl)
            s_mcc   = harmonic_metrics.get('mcc_with_refl', 0.0)
            s_h4    = harmonic_metrics.get('h4_mcc', 0.0)
            s_fbeta = test_metrics_with_refl.get('fbeta', 0.0)
            s_fpr   = test_metrics_with_refl.get('fpr', 0.0)
            print(
                f"Val E{epoch}: MCC={s_mcc:.4f} | "
                f"H4={s_h4:.4f} | "
                f"BAc={test_metrics_with_refl.get('balanced_accuracy', 0.0):.4f} | "
                f"Fbeta={s_fbeta:.4f} | "
                f"FPR={s_fpr:.4f}"
            )
            if _teacher_baseline:
                t_mcc   = _teacher_baseline.get('mcc_refl', float('nan'))
                t_h4    = _teacher_baseline.get('h4_mcc',   float('nan'))
                t_fbeta = _teacher_baseline.get('fbeta_refl', float('nan'))
                t_fpr   = _teacher_baseline.get('brier_refl', float('nan'))
                def _d(s, t): return f"{s - t:+.4f}" if (s == s and t == t) else "n/a"
                print(
                    f"vs Teacher:  MCC={t_mcc:.4f} ({_d(s_mcc, t_mcc)}) | "
                    f"H4={t_h4:.4f} ({_d(s_h4, t_h4)}) | "
                    f"Fbeta={t_fbeta:.4f} ({_d(s_fbeta, t_fbeta)})"
                )

            test_metrics = update_test_metrics_with_harmonic(test_metrics_with_refl.copy(), harmonic_metrics)

            # Track recall for Phase 3 RecallAdaptiveAlpha
            last_val_recall = test_metrics_with_refl.get('recall', None)

        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)

        eval_interval = int(getattr(args, 'eval_interval', 10))
        if bool(getattr(args, 'eval', False)) and eval_interval > 0 and epoch % eval_interval == 0:
            applied_eval_ema = ema_model is not None and not args.test
            if applied_eval_ema:
                ema_model.apply_shadow()
            run_eval_visualization(student_model, args, device, epoch, save_outputs=False)
            if applied_eval_ema:
                ema_model.restore()

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
                        'model_state_dict': student_model.state_dict(),
                        'model_config': _model_config,
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
                    'model_config': _model_config,
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
