from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.utils import one_hot

from torch_geometric.utils import scatter
from torch_scatter import scatter_add, scatter_max

class VoxelSampling:
    r"""Clusters points into fixed-sized voxels.
    Each cluster returned is a new point based on the mean of all points
    inside the given cluster.

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`pos`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        start: Optional[Union[float, List[float], Tensor]] = None,
        end: Optional[Union[float, List[float], Tensor]] = None,
    ) -> None:
        self.start = start
        self.end = end

    def __call__(
        self,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        reflectance: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        size: Optional[Tensor] = None,
    ):
        """
        Args:
            pos (Tensor): Point positions of shape [N, 3]
            batch (Tensor, optional): Batch indices of shape [N]
            reflectance (Tensor, optional): Reflectance values of shape [N]
            y (Tensor, optional): Label values of shape [N]
            size (Tensor, optional): Size of voxels

        Returns:
            pos_out (Tensor): Position tensor
            reflectance_out (Optional[Tensor]): Reflectance tensor if reflectance was provided
            batch_out (Tensor): Batch indices
            y_out (Optional[Tensor]): Label tensor if y was provided
        """
        num_nodes = pos.size(0)
        
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
        else:
            batch = batch.long()        

        c = torch_geometric.nn.voxel_grid(pos, size, batch, self.start, self.end)
        c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)

        pos_out = scatter(pos, c, dim=0, reduce='mean')
        batch_out = batch[perm].long()

        # Process reflectance only if provided
        reflectance_out = None
        if reflectance is not None:
            reflectance_out = scatter(reflectance, c, dim=0, reduce='mean')

        # Process y/labels only if provided
        y_out = None
        if y is not None:
            # Convert float labels to long for one_hot encoding
            # PyTorch Geometric's one_hot requires int64/long type
            y_long = y.long() if y.dtype != torch.long else y
            
            # Using torch_geometric approach for labels
            y_out = scatter(one_hot(y_long), c, dim=0, reduce='sum')
            y_out = y_out.argmax(dim=-1)

        # Return values conditionally based on what was provided
        result = [pos_out, reflectance_out, batch_out]
        if y is not None:
            result.append(y_out)
            
        return tuple(result)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'

class VoxelSamplingMax:
    r"""Clusters points into fixed-sized voxels.
    Each cluster returned is a new point with the highest reflectance value
    from all points inside the given cluster.

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`pos`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        start: Optional[Union[float, List[float], Tensor]] = None,
        end: Optional[Union[float, List[float], Tensor]] = None,
    ) -> None:
        self.start = start
        self.end = end

    def __call__(
        self,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        reflectance: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        size: Optional[Tensor] = None,
    ):
        """
        Args:
            pos (Tensor): Point positions of shape [N, 3]
            batch (Tensor, optional): Batch indices of shape [N]
            reflectance (Tensor, optional): Reflectance values of shape [N]
            y (Tensor, optional): Label values of shape [N]
            size (Tensor, optional): Size of voxels

        Returns:
            pos_out (Tensor): Position tensor
            reflectance_out (Optional[Tensor]): Reflectance tensor if reflectance was provided
            batch_out (Tensor): Batch indices
            y_out (Optional[Tensor]): Label tensor if y was provided
        """
        num_nodes = pos.size(0)
        
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
        else:
            batch = batch.long()        

        c = torch_geometric.nn.voxel_grid(pos, size, batch, self.start, self.end)
        
        if reflectance is None:
            c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)
            pos_out = scatter(pos, c, dim=0, reduce='mean')
            batch_out = batch[perm].long()
            
            # Process y/labels only if provided
            y_out = None
            if y is not None:
                # Convert float labels to long for one_hot encoding
                # PyTorch Geometric's one_hot requires int64/long type
                y_long = y.long() if y.dtype != torch.long else y
                
                # Using torch_geometric approach for labels
                y_out = scatter(one_hot(y_long), c, dim=0, reduce='sum')
                y_out = y_out.argmax(dim=-1)
                
            # Return values conditionally based on what was provided
            result = [pos_out, None, batch_out]
            if y is not None:
                result.append(y_out)
                
            return tuple(result)
        
        _, argmax = scatter_max(reflectance, c, dim=0)
        
        unique_clusters = torch.unique(c)
        
        max_indices = argmax[unique_clusters]
        
        pos_out = pos[max_indices]
        batch_out = batch[max_indices]
        reflectance_out = reflectance[max_indices]
        
        # Process y/labels only if provided
        y_out = None
        if y is not None:
            # For max sampling, take the y value at the same index as the max reflectance
            y_out = y[max_indices]
        
        # Return values conditionally based on what was provided
        result = [pos_out, reflectance_out, batch_out]
        if y is not None:
            result.append(y_out)
            
        return tuple(result)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'

class RandomSampling:
    r"""Randomly samples points with a defined ratio from each batch.

    Args:
        ratio (float): Ratio of points to keep (0.0-1.0).
        replace (bool, optional): Whether to sample with or without replacement.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        ratio: float = 0.5,
        replace: bool = False,
    ) -> None:
        self.ratio = ratio
        self.replace = replace

    def __call__(
        self,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        reflectance: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ):
        """
        Args:
            pos (Tensor): Point positions of shape [N, 3]
            batch (Tensor, optional): Batch indices of shape [N]
            reflectance (Tensor, optional): Reflectance values of shape [N]
            y (Tensor, optional): Label values of shape [N]

        Returns:
            pos_out (Tensor): Position tensor
            reflectance_out (Optional[Tensor]): Reflectance tensor if reflectance was provided
            batch_out (Tensor): Batch indices
            y_out (Optional[Tensor]): Label tensor if y was provided
        """
        num_nodes = pos.size(0)
        
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
        else:
            batch = batch.long()
        
        unique_batch, counts = torch.unique(batch, return_counts=True)
        num_batches = unique_batch.size(0)
        
        samples_per_batch = (counts * self.ratio).round().long()
        
        selected_indices = []
        
        for i in range(num_batches):
            batch_mask = batch == unique_batch[i]
            batch_indices = torch.nonzero(batch_mask).squeeze()
            
            if batch_indices.numel() == 0 or samples_per_batch[i] >= batch_indices.numel():
                selected_indices.append(batch_indices)
                continue
            
            perm = torch.randperm(batch_indices.numel(), device=pos.device)
            selected = batch_indices[perm[:samples_per_batch[i]]]
            selected_indices.append(selected)
        
        if len(selected_indices) > 0:
            selected_indices = torch.cat(selected_indices)
            
        pos_out = pos[selected_indices]
        batch_out = batch[selected_indices]
        
        reflectance_out = None
        if reflectance is not None:
            reflectance_out = reflectance[selected_indices]
            
        y_out = None
        if y is not None:
            y_out = y[selected_indices]
        
        # Return values conditionally based on what was provided
        result = [pos_out, reflectance_out, batch_out]
        if y is not None:
            result.append(y_out)
            
        return tuple(result)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(ratio={self.ratio}, replace={self.replace})'

