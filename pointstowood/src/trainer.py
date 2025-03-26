from src.model import Net
from src.augmentation import augmentations
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
from abc import ABC
from time import sleep
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
from src.loss import *
from torch.optim import AdamW
import warnings
import glob
import random
import wandb

# Configure PyTorch settings
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)

# Set random seed for reproducibility
seed = 141190
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def write_xyz_to_file(tensor: torch.Tensor, filename: str):
    tensor_np = tensor.cpu().numpy()    
    with open(filename, 'w') as f:
        for row in tensor_np:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")

class TrainingDataset(Dataset, ABC):
    def __init__(self, voxels, augmentation, mode, max_pts, device):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
        self.keys = sorted(glob.glob(os.path.join(voxels, '*.pt')))
        self.device = device
        self.max_pts = max_pts
        self.reflectance_index = 3
        self.label_index = 4
        self.augmentation = augmentation
        self.mode = mode

    def __len__(self):
        return len(self.keys) 

    def __getitem__(self, index):
        point_cloud = torch.load(self.keys[index], weights_only=True)
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)
        y = torch.as_tensor(point_cloud[:, self.label_index], dtype=torch.float)
        
        if self.augmentation:
            pos, reflectance, y = augmentations(pos, reflectance, y, self.mode)
        
        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        
        if torch.any(torch.isnan(reflectance)):
            print('nans in reflectance')
                
        data = Data(
            pos=pos,
            reflectance=reflectance,
            y=y,
            local_shift=local_shift,
            label_weighting=None
        )    
        return data

class ModelManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        adjusted_state_dict = OrderedDict()
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('module.'):
                key = key[7:]
            adjusted_state_dict[key] = value
        self.model.load_state_dict(adjusted_state_dict)
        return self.model

    def save_checkpoints(self, args, epoch):
        checkpoint_folder = os.path.join(args.wdir, 'checkpoints')
        if not os.path.isdir(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        file = checkpoint_folder + '/'f'epoch_{epoch}.pth'
        torch.save({'model_state_dict': self.model.state_dict()}, file)
        return True

    def save_best_model(self, stat, best_stat, save_path):
        if stat > best_stat:
            best_stat = stat
            torch.save({'model_state_dict': self.model.state_dict()}, save_path)
            print(f'Saving ', save_path)
        return best_stat

#########################################################################################################
#                                       SEMANTIC TRAINING FUNCTION                                      #
#                                       ==========================                                      #

def SemanticTraining(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project="pointstowood",
            config={
                "learning_rate": 5e-5,
                "batch_size": args.batch_size,
                "max_pts": args.max_pts,
                "augmentation": args.augmentation,
                "num_epochs": args.num_epochs
            }
        )
    
    # Initialize model and manager
    model = Net(num_classes=1, num_epochs=args.num_epochs).to(device)
    manager = ModelManager(model, device)
    print('Model contains', sum(p.numel() for p in model.parameters()), ' parameters')

    # Setup data loaders
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(
        TrainingDataset(voxels=args.trfile, augmentation=args.augmentation, mode='train', device=device, max_pts=args.max_pts),
        batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    test_loader = None
    if args.test:
        test_loader = DataLoader(
            TrainingDataset(voxels=args.tefile, augmentation=True, mode='test', device=device, max_pts=args.max_pts),
            batch_size=int(args.batch_size/2), shuffle=True, drop_last=True, 
            num_workers=num_workers, pin_memory=True
        )

    # Setup training components
    criterion = FocalLoss(reduction="mean", gamma=3.0, alpha=None, label_smoothing=None)
    #criterion = CyclicalFocalLoss(gamma_hc=4.0, gamma_lc=0.5, fc=4.0, num_epochs=args.num_epochs, alpha=None, label_smoothing = 0.2)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=args.num_epochs, 
                                                      pct_start=0.08, anneal_strategy='cos', div_factor=100)

    # Load or create model
    model_path = os.path.join(args.wdir, 'model', args.model)
    if os.path.isfile(model_path):
        try:
            manager.load_model(model_path)
        except KeyError:
            torch.save({'model_state_dict': model.state_dict()}, model_path)
    else:
        torch.save({'model_state_dict': model.state_dict()}, model_path)

    # Initialize tracking variables
    best_metrics = {
        'ba_train': 0.0, 'f1_train': 0.0,
        'ba_test': 0.0, 'f1_test': 0.0,
        'precision_test': 0.0, 'recall_test': 0.0
    }
    history = None

    def calculate_metrics(y_true, y_pred):
        """Calculate metrics for both wood and leaf, plus class ratio"""
        return {
            'precision': precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'bacc': balanced_accuracy_score(y_true, y_pred),
            'leaf_pr': precision_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0),
            'leaf_re': recall_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0),
            'wood_ratio': np.mean(y_true)
        }

    def process_batch(data, model, criterion, optimizer=None, is_training=True):
        """Process a single batch of data"""
        data = data.to(device)
        
        if is_training:
            optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                           enabled=True, dtype=torch.bfloat16):
            outputs = model(data)
            outputs = torch.clamp(outputs, min=-20.0, max=20.0)
            loss, gamma = criterion(outputs, data.y)

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(outputs) >= 0.50).type(torch.int64)
            metrics = calculate_metrics(data.y.cpu().numpy(), preds.cpu().numpy())
            
        return loss.item(), metrics, gamma if is_training else None

    for epoch in range(1, args.num_epochs + 1):
        print("\n" + "="*93)
        print(f"EPOCH {epoch}")
        
        # Training phase
        model.train()
        criterion.current_epoch = epoch
        train_stats = {
            'loss': 0.0, 
            'bacc': 0.0, 
            'precision': 0.0, 
            'recall': 0.0, 
            'f1': 0.0,
            'leaf_pr': 0.0,
            'leaf_re': 0.0,
            'wood_ratio': 0.0
        }
        valid_batches = 0
        
        with tqdm(total=len(train_loader), colour='white', ascii="░▒", 
                  bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
            for i, data in enumerate(train_loader):
                try:
                    loss, metrics, gamma = process_batch(data, model, criterion, optimizer, is_training=True)
                    
                    # Update running statistics
                    valid_batches += 1
                    train_stats['loss'] += loss
                    for k, v in metrics.items():
                        train_stats[k] = ((train_stats[k] * (valid_batches-1)) + v) / valid_batches
                    
                    # Update progress bar
                    pbar.set_description("Train")
                    pbar.set_postfix({
                        'Lo': f"{train_stats['loss']/valid_batches:.3f}",
                        'BAcc': f"{train_stats['bacc']:.3f}",
                        'F1': f"{train_stats['f1']:.3f}",
                        'W_Pr': f"{train_stats['precision']:.3f}",
                        'W_Re': f"{train_stats['recall']:.3f}",
                        'L_Pr': f"{train_stats['leaf_pr']:.3f}",
                        'L_Re': f"{train_stats['leaf_re']:.3f}",
                        'Ratio': f"{train_stats['wood_ratio']:.2f}",
                        'Ga': f"{gamma:.3f}"
                    })
                    pbar.update()
                    
                except RuntimeError as e:
                    print(f"\nSkipping batch due to error: {str(e)}")
                    continue

            lr_scheduler.step()

        # Testing phase
        if args.test:
            model.eval()
            test_stats = {
                'bacc': 0.0, 
                'precision': 0.0, 
                'recall': 0.0, 
                'f1': 0.0,
                'leaf_pr': 0.0,
                'leaf_re': 0.0,
                'wood_ratio': 0.0
            }
            valid_batches = 0
            
            with torch.no_grad(), tqdm(total=len(test_loader), colour='white', ascii="▒█",
                                     bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
                for j, data in enumerate(test_loader):
                    try:
                        _, metrics, _ = process_batch(data, model, criterion, is_training=False)
                        
                        # Accumulate metrics
                        valid_batches += 1
                        for k, v in metrics.items():
                            test_stats[k] = ((test_stats[k] * (valid_batches-1)) + v) / valid_batches
                        
                        # Update progress bar with running averages
                        pbar.set_description("Test")
                        pbar.set_postfix({
                            'Lo': f"{test_stats['loss']/valid_batches:.3f}" if 'loss' in test_stats else "N/A",
                            'BAcc': f"{test_stats['bacc']:.3f}",
                            'F1': f"{test_stats['f1']:.3f}",
                            'W_Pr': f"{test_stats['precision']:.3f}",
                            'W_Re': f"{test_stats['recall']:.3f}",
                            'L_Pr': f"{test_stats['leaf_pr']:.3f}",
                            'L_Re': f"{test_stats['leaf_re']:.3f}",
                            'Ratio': f"{test_stats['wood_ratio']:.2f}",
                        })
                        pbar.update()
                        
                    except Exception as e:
                        print(f"\nSkipping batch due to error: {str(e)}")
                        continue

                # No need for final averaging since we've been doing running averages
                if valid_batches == 0:
                    test_stats = None

        # Save history and checkpoints
        epoch_results = np.array([[
            epoch, optimizer.param_groups[0]["lr"],
            train_stats['loss']/len(train_loader),
            train_stats['bacc']/len(train_loader),
            train_stats['f1']/len(train_loader),
            train_stats['precision']/len(train_loader),
            train_stats['recall']/len(train_loader)
        ]])
        
        if args.test and test_stats is not None:
            epoch_results = np.append(epoch_results, [[
                test_stats['bacc']/len(test_loader),
                test_stats['f1']/len(test_loader),
                test_stats['precision']/len(test_loader),
                test_stats['recall']/len(test_loader)
            ]], axis=1)
        elif args.test:
            # Add zeros if no valid test batch was processed
            epoch_results = np.append(epoch_results, [[0.0, 0.0, 0.0, 0.0]], axis=1)
        
        history = epoch_results if epoch == 1 else np.vstack((history, epoch_results))
        np.savetxt(os.path.join(args.wdir, 'model', f"{os.path.splitext(args.model)[0]}_history.csv"), history)
        
        if epoch in args.checkpoints:
            manager.save_checkpoints(args, epoch)

        # Early stopping with consecutive decreases
        if args.stop_early and epoch > 10:
            if not hasattr(SemanticTraining, 'consec_decreases'):
                SemanticTraining.consec_decreases = 0
            
            current_acc = history[-1, 3]
            prev_acc = history[-2, 3]
            
            if current_acc < prev_acc:
                SemanticTraining.consec_decreases += 1
            else:
                SemanticTraining.consec_decreases = 0
                
            if SemanticTraining.consec_decreases >= 10:
                print(f"\nStopping early at epoch {epoch} - training accuracy decreased for {SemanticTraining.consec_decreases} consecutive epochs")
                print(f"Best accuracy was {max(history[:, 3]):.4f} at epoch {np.argmax(history[:, 3]) + 1}")
                break

        # Save best models - testing
        if args.test and test_stats is not None and epoch > int(args.num_epochs*0.25):
            test_precision = test_stats['precision']/len(test_loader)
            test_recall = test_stats['recall']/len(test_loader)
            #if test_precision >= 0.99 * test_recall:
            best_metrics['ba_test'] = manager.save_best_model(
                test_stats['bacc']/len(test_loader),
                best_metrics['ba_test'],
                os.path.join(args.wdir, 'model', f'ba-{args.model}'))
            best_metrics['f1_test'] = manager.save_best_model(
                test_stats['f1']/len(test_loader),
                best_metrics['f1_test'],
                os.path.join(args.wdir, 'model', f'f1-{args.model}'))
            best_metrics['precision_test'] = manager.save_best_model(
                test_precision,
                best_metrics['precision_test'],
                os.path.join(args.wdir, 'model', f'precision-{args.model}'))

        # Log to wandb if enabled
        if args.wandb:
            wandb_metrics = {
                "Epoch": epoch,
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "Loss": np.around(train_stats['loss']/valid_batches, 4),
                "Accuracy": np.around(train_stats['bacc'], 4),
                "Precision": np.around(train_stats['precision'], 4),
                "Recall": np.around(train_stats['recall'], 4),
                "F1": np.around(train_stats['f1'], 4)
            }
            
            if args.test:
                wandb_metrics.update({
                    "Test F1": np.around(test_stats['f1'], 4),
                    "Test Accuracy": np.around(test_stats['bacc'], 4),
                    "Test Precision": np.around(test_stats['precision'], 4),
                    "Test Recall": np.around(test_stats['recall'], 4)
                })
            
            wandb.log(wandb_metrics)

        # Save final model at last epoch
        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            torch.save({'model_state_dict': model.state_dict()}, model_path)
