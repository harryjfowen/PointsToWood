from src.model import Net
from src.dataset import create_inference_loader
import os
import sys
import pandas as pd
import numpy as np
from pykdtree.kdtree import KDTree
from tqdm.auto import tqdm
import torch
from src.io import save_file
from collections import OrderedDict
from numba import jit, prange

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.setrecursionlimit(10 ** 8)        
        
from collections import OrderedDict
def load_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    adjusted_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            key = key[7:]
        adjusted_state_dict[key] = value
    model.load_state_dict(adjusted_state_dict, strict=False)
    return model
    
class PointCloudClassifier:
    """Aggregate KNN predictions into a final label per original point.

    Modes (mutually exclusive, priority top→bottom):
      1. *max_probability*  – choose the most confident prediction in the
         neighborhood (furthest from 0.5).
      2. *any_wood* (value ≠ 1) – if *any* probability ≥ any_wood, label as wood
         irrespective of aggregate probability.
      3. Default – take the median probability and compare against *is_wood*.
    """

    def __init__(self, is_wood: float, any_wood: float, max_probability: bool = False):
        self.is_wood = is_wood
        self.any_wood = any_wood
        self.max_probability = max_probability

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_median_threshold(nbr_classification, labels, is_wood):
        """Median probability compared with *is_wood* threshold."""
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            median_prob = np.median(nbr_classification[i, :, -1])
            labels[i, 1] = median_prob
            labels[i, 0] = 1 if median_prob >= is_wood else 0
        return labels

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_any_wood(nbr_classification, labels, any_wood):
        """Label wood if *any* probability ≥ any_wood, else leaf."""
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            probs = nbr_classification[i, :, -1]
            labels[i, 1] = np.median(probs)
            labels[i, 0] = 1 if np.any(probs >= any_wood) else 0
        return labels

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_argmax(nbr_classification, labels):
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            probs = nbr_classification[i, :, -1]
            conf_idx = np.argmax(np.abs(probs - 0.5))
            conf_prob = probs[conf_idx]
            labels[i, 1] = conf_prob
            labels[i, 0] = nbr_classification[i, conf_idx, -2]
        return labels

    def collect_predictions(self, classification, original):
        original = original.drop(columns=[c for c in original.columns if c in ['pred', 'pwood', 'pleaf']])
        
        indices_file = os.path.join('nbrs.npy')
        
        if os.path.exists(indices_file):
            indices = np.load(indices_file)
        else:
            kd_tree = KDTree(classification[:, :3])
            _, indices = kd_tree.query(original.values[:, :3], k = 8 if self.any_wood != 1 else 16)

        labels = np.zeros((original.shape[0], 2))
        
        if self.max_probability:
            labels = self._labels_argmax(classification[indices], labels)
        elif self.any_wood != 1:
            labels = self._labels_any_wood(classification[indices], labels, self.any_wood)
        else:
            labels = self._labels_median_threshold(classification[indices], labels, self.is_wood)
            
        original.loc[:, ['pred', 'pwood']] = labels
        return original

def SemanticSegmentation(args):
    '''
    Setup Multi GPU processing. 
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    Setup model. 
    '''
    model = Net(num_classes=1).to(device)

    try:
        load_model(os.path.join(args.wdir,'model',args.model), model, device)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    
    '''
    Setup data loader. 
    '''
    test_loader, test_dataset = create_inference_loader(args, device)


    '''
    Initialise model
    '''
    model.eval()
    output_list = []

    with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc="Inference") as pbar:
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(data)
                outputs = torch.nan_to_num(outputs)
            
            probs = torch.sigmoid(outputs).to(device)
            
            preds = (probs >= args.is_wood).type(torch.int64).cpu()
            preds = np.expand_dims(preds, axis=1)

            batches = np.unique(data.batch.cpu())
            pos = data.pos.cpu().numpy()
            probs_2d = np.expand_dims(probs.detach().cpu().numpy(), axis=1)  
            output = np.concatenate((pos, preds, probs_2d), axis=1)
            
            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch : 3 + (3 * batch)]
                output_list.append(outputb)
            
            pbar.update(1)
    
    classified_pc = np.vstack(output_list)

    
    '''
    Choosing most confident labels using nearest neighbour search. 
    '''  
    if args.verbose: print("Spatially aggregating prediction probabilites and labels...")

    classifier = PointCloudClassifier(
        is_wood=args.is_wood,
        any_wood=args.any_wood,
        max_probability=args.max_probability,
    )
    args.pc = classifier.collect_predictions(classified_pc, args.pc)

    '''
    Save final classified point cloud. 
    '''
    headers = list(dict.fromkeys(args.headers + ['pred', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields=headers, verbose=False)    
    
    return args
