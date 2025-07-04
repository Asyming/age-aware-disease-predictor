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
    
class MixDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, feat_path, sample_ids, ages, labels, balanced_sampling=False):
        self.sample_ids = sample_ids
        self.ages = ages
        self.feat_path = feat_path
        self.balanced_sampling = balanced_sampling
        
        if isinstance(labels[0], str):
            parsed_labels = []
            for label in labels:
                label_str = label.strip('[]')
                ad_label, ms_label = [int(x) for x in label_str.split(',')]
                parsed_labels.append([ad_label, ms_label])
            self.labels = torch.tensor(parsed_labels).float()
        else:
            self.labels = torch.from_numpy(np.array(labels)).float()
        
        if self.balanced_sampling:
            self.n_samples = 10000000
            # 有阳性：[1,1], [0,1], [1,0] 
            # 双阴性：[0,0]
            has_positive = (self.labels[:, 0] == 1) | (self.labels[:, 1] == 1)
            is_double_negative = (self.labels[:, 0] == 0) & (self.labels[:, 1] == 0)
            
            self.positive_indices = torch.where(has_positive)[0]
            self.negative_indices = torch.where(is_double_negative)[0]
            
            label_combinations = {}
            for i in range(len(self.labels)):
                key = f"[{int(self.labels[i][0])},{int(self.labels[i][1])}]"
                label_combinations[key] = label_combinations.get(key, 0) + 1
            
            neg_ages = ages[self.negative_indices]
            self.age_mean = float(np.mean(neg_ages))
            self.age_std = float(np.std(neg_ages))
        else:
            self.n_samples = len(self.labels)
            
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.balanced_sampling:
            is_positive = random.choice([True, False])
            
            if is_positive and len(self.positive_indices) > 0:
                index = random.choice(self.positive_indices)
            elif not is_positive and len(self.negative_indices) > 0:
                index = random.choice(self.negative_indices)
            else:
                index = random.randint(0, len(self.labels) - 1)
        
        sample_id = self.sample_ids[index]
        label = self.labels[index]
        age = self.ages[index]
        feat = np.load(f'{self.feat_path}/{sample_id}.npy').astype(np.float32)
        
        return {"feat": feat, "label": label, "age": age}