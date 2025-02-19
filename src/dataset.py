import numpy as np
import torch
import random
class Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self,feat_path, sample_ids, ages, labels, balanced_sampling=False):
        self.sample_ids = sample_ids
        self.labels = torch.from_numpy(labels).float()
        self.ages = ages
        self.feat_path = feat_path
        self.balanced_sampling = balanced_sampling
        if self.balanced_sampling:
            self.n_samples = 10000000
            self.classes = torch.unique(self.labels)
            self.class_indices = {cls.item(): torch.where(self.labels == cls)[0] for cls in self.classes}
        else:
            self.n_samples = len(self.labels)    
    def __len__(self):
        return self.n_samples

    def __init__(self, feat_path, sample_ids, ages, labels, balanced_sampling=False):
        self.sample_ids = sample_ids
        self.labels = torch.from_numpy(labels).float()
        self.ages = ages
        self.feat_path = feat_path
        self.balanced_sampling = balanced_sampling
        
        if self.balanced_sampling:
            self.n_samples = 10000000
            self.classes = torch.unique(self.labels)
            self.class_indices = {cls.item(): torch.where(self.labels == cls)[0] for cls in self.classes}
            
            neg_ages = ages[labels == 0]
            self.age_mean = float(np.mean(neg_ages))
            self.age_std = float(np.std(neg_ages))
        else:
            self.n_samples = len(self.labels)

    def __getitem__(self, index):
        if self.balanced_sampling:
            cls = random.choice(self.classes)
            sample_indices = self.class_indices[cls.item()]
            index = random.choice(sample_indices)
        sample_id = self.sample_ids[index]
        label = self.labels[index]
        age = self.ages[index]
        feat = np.load(f'{self.feat_path}/{sample_id}.npy').astype(np.float32)
        return {"feat": feat, "label": label, "age": age}