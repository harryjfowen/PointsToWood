import numpy as np
import torch
import os
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from collections import OrderedDict


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    iou_wood = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    iou_leaf = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0.0
    miou = (iou_wood + iou_leaf) / 2

    accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    Fbeta = fbeta_score(y_true, y_pred, beta=0.90, average='binary', zero_division=0)
    precision_wood = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_wood = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    return accuracy, precision, recall, f1, miou, precision_wood, recall_wood, Fbeta


class MetricsTracker:
    """Tracks and accumulates metrics during training/testing."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.loss = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.miou = 0.0
        self.precision_wood = 0.0
        self.recall_wood = 0.0
        self.fbeta = 0.0
        self.num_batches = 0
    
    def update(self, loss, outputs, targets, edge_scores=None):
        """Update metrics with new batch results."""
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.50).int()
            y_true = targets.cpu().numpy().astype(int)
            y_pred = preds.cpu().numpy().astype(int)
            
            acc, prec, rec, f1, miou, prw, rew, fbeta = calculate_metrics(y_true, y_pred)
            
            self.loss += loss.item()
            self.accuracy += acc
            self.precision += prec
            self.recall += rec
            self.f1 += f1
            self.miou += miou
            self.precision_wood += prw
            self.recall_wood += rew
            self.fbeta += fbeta
            self.num_batches += 1
    
    def update_with_smoothing(self, loss, outputs, targets, pos, batch, edge_scores=None):
        """Update metrics with KNN-smoothed predictions for testing."""
        with torch.no_grad():
            probs = torch.sigmoid(outputs).detach()
            
            from torch_geometric.nn import knn
            import torch_scatter
            row, col = knn(pos[:, :3], pos[:, :3], 8, batch, batch)
            idx = row.view(-1)  
            conf = torch.abs(probs - 0.5) + 1e-3
            w_sum = torch_scatter.scatter_add(conf[col] * probs[col], idx, dim=0)
            w_cnt = torch_scatter.scatter_add(conf[col], idx, dim=0)
            smoothed = (w_sum / w_cnt).clamp(0, 1) 
            
            preds = (smoothed >= 0.50).type(torch.int64).detach()
            y_true = targets.cpu().numpy().astype(int)
            y_pred = preds.cpu().numpy().astype(int)
            
            acc, prec, rec, f1, miou, prw, rew, fbeta = calculate_metrics(y_true, y_pred)
            
            self.loss += loss.item()
            self.accuracy += acc
            self.precision += prec
            self.recall += rec
            self.f1 += f1
            self.miou += miou
            self.precision_wood += prw
            self.recall_wood += rew
            self.fbeta += fbeta
            self.num_batches += 1
    
    def get_averages(self):
        """Get average metrics across all batches."""
        if self.num_batches == 0:
            return {k: 0.0 for k in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'miou', 'precision_wood', 'recall_wood', 'fbeta']}
        
        return {
            'loss': self.loss / self.num_batches,
            'accuracy': self.accuracy / self.num_batches,
            'precision': self.precision / self.num_batches,
            'recall': self.recall / self.num_batches,
            'f1': self.f1 / self.num_batches,
            'miou': self.miou / self.num_batches,
            'precision_wood': self.precision_wood / self.num_batches,
            'recall_wood': self.recall_wood / self.num_batches,
            'fbeta': self.fbeta / self.num_batches
        }


class ModelManager:
    """Handles model saving, loading, and checkpoint management."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def load_model(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        adjusted_state_dict = OrderedDict()
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('module.'):
                key = key[7:]
            adjusted_state_dict[key] = value
        self.model.load_state_dict(adjusted_state_dict)
        return self.model

    def save_checkpoints(self, args, epoch):
        """Save checkpoint at specified epoch."""
        checkpoint_folder = os.path.join(args.wdir, 'checkpoints')
        if not os.path.isdir(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        file = checkpoint_folder + '/' + f'epoch_{epoch}.pth'
        torch.save({'model_state_dict': self.model.state_dict()}, file)
        return True

    def save_best_model(self, stat, best_stat, save_path):
        """Save model if current stat is better than best."""
        if stat > best_stat:
            best_stat = stat
            torch.save({'model_state_dict': self.model.state_dict()}, save_path)
            print(f'Saving {save_path}')
        return best_stat


class HistoryLogger:
    """Handles training history logging and saving."""
    
    def __init__(self, args):
        self.args = args
        self.history = None
    
    def log_epoch(self, epoch, lr, train_metrics, test_metrics=None):
        """Log epoch results to history."""
        epoch_results = np.array([[
            epoch, lr,
            train_metrics['loss'],
            train_metrics['accuracy'],
            train_metrics['f1'],
            train_metrics['precision'],
            train_metrics['recall']
        ]])

        if test_metrics:
            epoch_results = np.append(epoch_results, [[
                test_metrics['accuracy'],
                test_metrics['f1'],
                test_metrics['precision'],
                test_metrics['recall']
            ]], axis=1)
        
        if self.history is None:
            self.history = epoch_results
        else:
            self.history = np.vstack((self.history, epoch_results))
        
        self.save_history()
    
    def save_history(self):
        """Save training history to file."""
        try:
            history_path = os.path.join(
                self.args.wdir, 'model', 
                os.path.splitext(self.args.model)[0] + "_history.csv"
            )
            np.savetxt(history_path, self.history)
            print("Saved training history successfully.")
        except OSError:
            backup_path = os.path.join(
                self.args.wdir, 'model', 
                os.path.splitext(self.args.model)[0] + "_history_backup.csv"
            )
            np.savetxt(backup_path, self.history)
    
    def get_best_epoch(self, metric_idx=3):
        """Get epoch with best metric (default: accuracy)."""
        if self.history is None or len(self.history) == 0:
            return 0, 0.0
        best_idx = np.argmax(self.history[:, metric_idx])
        return int(self.history[best_idx, 0]), self.history[best_idx, metric_idx]


class WandbLogger:
    """Handles Weights & Biases logging."""
    
    def __init__(self, args):
        self.args = args
        self.wandb = None
        if args.wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                project="PointsToWood", 
                config={
                    "architecture": "pointnet++",
                    "dataset": "high resolution 2 & 4 m voxels",
                    "epochs": args.num_epochs,
                }
            )
    
    def log_epoch(self, epoch, lr, train_metrics, test_metrics=None):
        """Log epoch metrics to wandb."""
        if not self.wandb:
            return
        
        log_dict = {
            "Epoch": epoch,
            "Learning Rate": lr,
            "Loss": np.around(train_metrics['loss'], 4),
            "Accuracy": np.around(train_metrics['accuracy'], 4),
            "Precision": np.around(train_metrics['precision'], 4),
            "Recall": np.around(train_metrics['recall'], 4),
            "F1": np.around(train_metrics['f1'], 4),
            "mIoU": np.around(train_metrics['miou'], 4),
        }
        
        if test_metrics:
            log_dict.update({
                "Test Accuracy": np.around(test_metrics['accuracy'], 4),
                "Test Precision": np.around(test_metrics['precision'], 4),
                "Test Recall": np.around(test_metrics['recall'], 4),
                "Test F1": np.around(test_metrics['f1'], 4),
                "Test Fbeta": np.around(test_metrics['fbeta'], 4),
                "Test mIoU": np.around(test_metrics['miou'], 4),
                "Test Precision_wood": np.around(test_metrics['precision_wood'], 4),
                "Test Recall_wood": np.around(test_metrics['recall_wood'], 4)
            })
        
        self.wandb.log(log_dict) 