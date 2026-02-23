import torch
import numpy as np
from mb.pandas.dfload import load_any_df
import mb.pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List,Optional

class FoodDataset(Dataset):
    '''
    Dataset class for loading food features, labels, and descriptions.
    Good upto 10M samples. For larger datasets, consider using a more scalable data loading approach.
    Args:
        df: A pandas DataFrame containing 'feature_path', 'label', and 'description' columns.
    Output:
        Tuples of (feature, label, text) for each sample.
    '''
    def __init__(self, 
                 df: pd.DataFrame, 
                 text_column: str | None = 'description'):
        self.df = df
        self.paths = df['feature_path'].values
        self.labels = df['label'].values
        self.texts = df[text_column].values if text_column is not None else None

    def __len__(self):
        return len(self.df)
    
    def __repr__(self):
        return f"FoodDataset(num_samples={len(self)})"
        
    def __getitem__(self, idx):

        feature = np.load(self.paths[idx], mmap_mode='r') ## Use memory-mapped mode to avoid loading the entire file into memory
        feature = torch.from_numpy(feature).float()
        
        label = self.labels[idx]
        text = self.texts[idx] if self.texts is not None else None
        
        return feature, label, text
    
def get_dist_loader(df, rank, world_size, batch_size=64):
    dataset = FoodDataset(df)
    
    #Sampler for each GPU 
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,      # Tune this based on your CPU cores
        pin_memory=True,    # Essential for fast GPU transfer
        prefetch_factor=2   # Pre-loads the next batches
    )
    return loader