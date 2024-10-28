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
from src.cosine_scheduler import CosineAnnealingWarmupRestarts
import warnings
import copy
import glob
from torch.cuda.amp import GradScaler, autocast
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)

seed = 141190
torch.manual_seed(seed)

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

        point_cloud = torch.load(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)
        y = torch.as_tensor(point_cloud[:, self.label_index], dtype=torch.float)
        if self.augmentation:
            pos, reflectance, y = augmentations(pos, reflectance, y, self.mode)
        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()
        if torch.any(torch.isnan(reflectance)):
            print('nans in relfectance')
        data = Data(pos=pos, reflectance=reflectance, y=y, sf=scaling_factor)
        return data

class ModelManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        adjusted_state_dict = OrderedDict()
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('module.'):
                key = key[7:]
            adjusted_state_dict[key] = value
        self.model.load_state_dict(adjusted_state_dict)
        return self.model

    def save_checkpoints(self, args, epoch):
        checkpoint_folder = os.path.join(args.wdir,'checkpoints')
        if not os.path.isdir(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        file = checkpoint_folder + '/'f'epoch_{epoch}.pth'
        torch.save({'model_state_dict': self.model.state_dict()}, file)
        return True

    def save_best_model(self, stat, best_stat, save_path):
        if stat > best_stat:
            best_stat = stat
            torch.save({'model_state_dict': self.model.state_dict()},  save_path)
            print(f'Saving ', save_path)
        return best_stat

#########################################################################################################
#                                       SEMANTIC TRAINING FUNCTION                                      #
#                                       ==========================                                      #

def SemanticTraining(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    torch.autograd.set_detect_anomaly(True)

    if args.wandb:
        import wandb
        wandb.init(project="PointsToWood", config={"architecture": "pointnet++","dataset": "high resolution 2 & 4 m voxels","epochs": args.num_epochs,})
        
    
    model = Net(num_classes=1).to(device)
    print('Model contains', sum(p.numel() for p in model.parameters()), ' parameters')

    train_dataset = TrainingDataset(voxels=args.trfile, augmentation=args.augmentation, mode='train', device=device, max_pts=args.max_pts)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=32, sampler = None, pin_memory=True)

    if args.test:
        test_dataset = TrainingDataset(voxels=args.tefile, augmentation=args.augmentation, mode='test', device=device, max_pts=args.max_pts)
        test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=True, drop_last=True, num_workers=32, sampler=None, pin_memory=True)

    criterion = Poly1FocalLoss(reduction="mean", gamma = 2.0, alpha = None, label_smoothing = 0.1)

    if args.tune: 
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, weight_decay=1e-2)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.num_epochs//5, cycle_mult=1.0, max_lr=1e-6, min_lr=1e-8, warmup_steps=5, gamma=0.5)
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=args.num_epochs, pct_start=0.05, anneal_strategy='cos', div_factor = 100)
            
    manager = ModelManager(model, device)

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

    def log_history(args,history):
        try:
            history = np.savetxt(os.path.join(args.wdir, 'model', os.path.splitext(args.model)[0] + "_history.csv"), history)
            print("Saved training history successfully.")

        except OSError:
            history = np.savetxt(os.path.join(args.wdir, 'model', os.path.splitext(args.model)[0] + "_history_backup.csv"), history)
            pass

    lr_list = []
    gamma = 0.0,0.0
    best_ba_train, best_f1_train, best_ba_test, best_f1_test, best_precision_test = 0.0, 0.0, 0.0, 0.0, 0.0

    scaler = GradScaler(init_scale=512)

    for epoch in range(1, args.num_epochs + 1):

        model.train()
        accumulation_steps = 1

        train_loss, train_accuracy, train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        sleep(0.1)
        print("\n=============================================================================================")
        print("EPOCH ", epoch)

        with tqdm(total=len(train_loader), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, (data) in enumerate(train_loader):
                
                data = data.to(device)
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                try:
                    model_params_before = copy.deepcopy(model.state_dict())

                    with autocast():
                        outputs = model(data)
                        loss, gamma = criterion(outputs, data.y)
                        loss = loss / accumulation_steps 

                    scaler.scale(loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                except Exception as e:
                    print(f"Skipping a batch due to an error: {e}")
                    model.load_state_dict(model_params_before)
                    optimizer.zero_grad()
                    continue

                with torch.no_grad():
                    probs = torch.sigmoid(outputs).detach()
                    preds = (probs >= 0.50).type(torch.int64).detach()

                train_loss += loss.item()
                train_precision += precision_score(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)
                train_recall += recall_score(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)
                train_accuracy += balanced_accuracy_score(data.y.cpu(), preds.cpu(), sample_weight=None)
                train_f1 += f1_score(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)

                lr_list.append(optimizer.param_groups[0]["lr"])
                tepoch.set_description(f"Train")
                tepoch.update()
                tepoch.set_postfix({
                    'Lr': optimizer.param_groups[0]["lr"],
                    'Lo': np.around(train_loss / (i + 1), 5),
                    'Ac': np.around(train_accuracy / (i + 1), 3),
                    'Pr': np.around(train_precision / (i + 1), 3),
                    'Re': np.around(train_recall / (i + 1), 3),
                    'F1': np.around(train_f1 / (i + 1), 3),
                    'Ga': np.around(gamma, 3)
                })

            tepoch.close()

            lr_scheduler.step()

        test_accuracy, test_precision, test_recall, test_f1 = 0.0, 0.0, 0.0, 0.0

        if args.test:

            model.eval()
            sleep(0.1) 

            with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
                with torch.no_grad():
                    for j, data in enumerate(test_loader):

                        data.y = data.y.to(device)

                        with torch.no_grad():  
                            outputs = model(data.to(device))
                            probs = torch.sigmoid(outputs).detach()
                            preds = (probs >= 0.50).type(torch.int64).detach()                        

                        test_precision += precision_score(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)
                        test_recall += recall_score(data.y.cpu(), preds.cpu(),average='binary', zero_division=0)
                        test_accuracy += balanced_accuracy_score(data.y.cpu(), preds.cpu(), sample_weight=None)
                        test_f1 += f1_score(data.y.cpu(), preds.cpu(),average='binary', zero_division=0)

                        lr_list.append(optimizer.param_groups[0]["lr"])
                        tepoch.set_description(f"Test")
                        tepoch.update()
                        tepoch.set_postfix({
                            'Ac': np.around(test_accuracy / (j + 1), 3),
                            'Pr': np.around(test_precision / (j + 1), 3),
                            'Re': np.around(test_recall / (j + 1), 3),
                            'F1': np.around(test_f1 / (j + 1), 3),
                        })

                    tepoch.close()
            
        epoch_results = np.array([[epoch, optimizer.param_groups[0]["lr"],
                                   train_loss/len(train_loader),
                                   train_accuracy/len(train_loader),
                                   train_f1/len(train_loader),
                                   train_precision/len(train_loader),
                                   train_recall/len(train_loader)]])

        if args.test:
            epoch_results = np.append(epoch_results, [[test_accuracy / len(test_loader),
                                                       test_f1 / len(test_loader),
                                                       test_precision / len(test_loader),
                                                       test_recall / len(test_loader)]], axis=1)
        
        if epoch == 1:
            history = epoch_results
        else:
            history = np.vstack((history,epoch_results))

        log_history(args,history)

        if epoch in args.checkpoints:
            manager.save_checkpoints(args, epoch)
        
        if args.stop_early:
            consec_decreases = 0  
            if epoch > 10: 
                current_acc = history[-1, 3] 
                prev_acc = history[-2, 3]     
                
                if current_acc < prev_acc:
                    consec_decreases += 1
                else:
                    consec_decreases = 0  
                    
                if consec_decreases >= 10: 
                    print(f"\nStopping early at epoch {epoch} - training accuracy decreased for {consec_decreases} consecutive epochs")
                    print(f"Best accuracy was {max(history[:, 3]):.4f} at epoch {np.argmax(history[:, 3]) + 1}")
                    break

        if epoch > int(args.num_epochs*0.10) and not args.test:
            best_ba_train = manager.save_best_model(train_accuracy/len(train_loader), best_ba_train, os.path.join(args.wdir,'model','ba-' + os.path.basename(args.model)))
            best_f1_train = manager.save_best_model(train_f1/len(train_loader), best_f1_train, os.path.join(args.wdir,'model','f1-' + os.path.basename(args.model)))
        
        if args.test and epoch > int(args.num_epochs*0.5):
            best_ba_test = manager.save_best_model(test_accuracy/len(test_loader), best_ba_test, os.path.join(args.wdir,'model','ba-' + os.path.basename(args.model)))
            best_f1_test = manager.save_best_model(test_f1/len(test_loader), best_f1_test, os.path.join(args.wdir,'model','f1-' + os.path.basename(args.model)))
            best_precision_test = manager.save_best_model(test_precision/len(test_loader), best_precision_test, os.path.join(args.wdir,'model','precision-' + os.path.basename(args.model)))

        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.wdir,'model',args.model))

        t_acc =  test_accuracy / len(test_loader) if args.test else 0.0
        t_f1 =  test_f1 / len(test_loader) if args.test else 0.0

        if args.wandb:
            wandb.log({"Epoch": epoch, 
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "Loss": np.around(train_loss/len(train_loader), 4),
                "Accuracy": np.around(train_accuracy/len(train_loader), 4),
                "Precision": np.around(train_precision/len(train_loader), 4),
                "Recall": np.around(train_recall/len(train_loader), 4),
                "F1": np.around(train_f1/len(train_loader), 4),
                "Test F1": np.around(t_f1, 4),
                "Test Accuracy": np.around(t_acc, 4)})
