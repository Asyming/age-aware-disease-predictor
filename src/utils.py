import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/zisen/ageaware"))
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.teacher_models import *

def set_random_seed(seed=22):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Set seed for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(batch):
    """Collate function for DataLoader"""
    feats = torch.from_numpy(np.stack([sample['feat'] for sample in batch])).float()
    labels = torch.FloatTensor([sample['label'] for sample in batch]).reshape(-1, 1)
    ages = torch.FloatTensor([sample['age'] for sample in batch]).reshape(-1, 1)
    return feats, labels, ages

def collate_fn_mix(batch):
    """Collate function for multi-label DataLoader"""
    feats = torch.from_numpy(np.stack([sample['feat'] for sample in batch])).float()
    labels = torch.stack([sample['label'] for sample in batch])
    ages = torch.FloatTensor([sample['age'] for sample in batch]).reshape(-1, 1)
    return feats, labels, ages

def split_by_age(labels, ages, age_threshold, n_splits=6):
    """Create train/val/test splits by age"""
    seeds = [20 + i for i in range(n_splits)]
    splits = []
    pos_indexs = np.where(labels == 1)[0]
    neg_indexs = np.where(labels == 0)[0]
    
    for seed in seeds:
        np.random.seed(seed)

        np.random.shuffle(pos_indexs)
        train_pos_size = int(len(pos_indexs) * 0.8)
        train_pos_data = pos_indexs[:train_pos_size]
        remaining_pos = pos_indexs[train_pos_size:]

        np.random.shuffle(neg_indexs)
        train_neg_size = int(len(neg_indexs) * 0.8)
        train_neg_data = neg_indexs[:train_neg_size]
        remaining_neg = neg_indexs[train_neg_size:]
        high_age_neg = remaining_neg[ages[remaining_neg] >= age_threshold]
        
        val_pos_size = len(remaining_pos) // 2
        val_pos_data = remaining_pos[:val_pos_size]
        test_pos_data = remaining_pos[val_pos_size:]
        val_neg_size = len(high_age_neg) // 2
        val_neg_data = high_age_neg[:val_neg_size]
        test_neg_data = high_age_neg[val_neg_size:]
        
        train_indices = np.concatenate((train_pos_data, train_neg_data))
        val_indices = np.concatenate((val_pos_data, val_neg_data))
        test_indices = np.concatenate((test_pos_data, test_neg_data))
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        splits.append((train_indices, val_indices, test_indices))
    
    return splits

def split_by_age_mix(labels, ages, age_threshold, n_splits=6):
    seeds = [20 + i for i in range(n_splits)]
    splits = []
    if isinstance(labels[0], str):
        parsed_labels = []  
        for label in labels:
            label_str = label.strip('[]')
            ad_label, ms_label = [int(x) for x in label_str.split(',')]
            parsed_labels.append([ad_label, ms_label])
        labels = np.array(parsed_labels)
    else:
        labels = np.array(labels)
    ages = np.array(ages)

    has_positive = (labels[:, 0] == 1) | (labels[:, 1] == 1)
    positive_indices = np.where(has_positive)[0]

    is_double_negative = (labels[:, 0] == 0) & (labels[:, 1] == 0)
    double_negative_indices = np.where(is_double_negative)[0]
    for seed in seeds:
        np.random.seed(seed)

        pos_indices_shuffled = positive_indices.copy()
        np.random.shuffle(pos_indices_shuffled)
        train_pos_size = int(len(pos_indices_shuffled) * 0.8)
        train_pos_data = pos_indices_shuffled[:train_pos_size]
        remaining_pos = pos_indices_shuffled[train_pos_size:]
        val_pos_size = len(remaining_pos) // 2
        val_pos_data = remaining_pos[:val_pos_size]
        test_pos_data = remaining_pos[val_pos_size:]

        double_neg_shuffled = double_negative_indices.copy()
        np.random.shuffle(double_neg_shuffled)
        train_neg_size = int(len(double_neg_shuffled) * 0.8)
        train_neg_data = double_neg_shuffled[:train_neg_size] 
        remaining_neg = double_neg_shuffled[train_neg_size:]
        high_age_neg = remaining_neg[ages[remaining_neg] >= age_threshold]
        val_neg_size = len(high_age_neg) // 2
        val_neg_data = high_age_neg[:val_neg_size]
        test_neg_data = high_age_neg[val_neg_size:]
        train_indices = np.concatenate((train_pos_data, train_neg_data))
        val_indices = np.concatenate((val_pos_data, val_neg_data))
        test_indices = np.concatenate((test_pos_data, test_neg_data))
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        splits.append((train_indices, val_indices, test_indices))
    
    return splits

def check_data(info_df, train_indices, val_indices, test_indices, run_id, age_threshold):
    print(f"\nSplit {run_id} Data Distribution:")
    print(f"Train set: {len(train_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[train_indices, 'label'] == 1)}/"
        f"{np.sum(info_df.loc[train_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[train_indices, 'label'] == 1):.3f}")
    print(f"\nValidation set: {len(val_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[val_indices, 'label'] == 1)}/"
        f"{np.sum(info_df.loc[val_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[val_indices, 'label'] == 1):.3f}")
    print(f"\nTest set (age >= {age_threshold}): {len(test_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[test_indices, 'label'] == 1)}/"
        f"{np.sum(info_df.loc[test_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[test_indices, 'label'] == 1):.3f}")

def check_data_mix(info_df, train_indices, val_indices, test_indices, run_id, age_threshold):
    """Check data distribution for multi-label dataset"""
    print(f"\nSplit {run_id} Data Distribution:")
    
    def parse_labels(labels):
        if isinstance(labels.iloc[0], str):
            parsed = []
            for label in labels:
                label_str = label.strip('[]')
                ad_label, ms_label = [int(x) for x in label_str.split(',')]
                parsed.append([ad_label, ms_label])
            return np.array(parsed)
        return np.array(labels.tolist())
    
    train_labels = parse_labels(info_df.loc[train_indices, 'label'])
    val_labels = parse_labels(info_df.loc[val_indices, 'label'])
    test_labels = parse_labels(info_df.loc[test_indices, 'label'])
    
    train_pos = np.sum((train_labels[:, 0] == 1) | (train_labels[:, 1] == 1))
    train_neg = np.sum((train_labels[:, 0] == 0) & (train_labels[:, 1] == 0))
    val_pos = np.sum((val_labels[:, 0] == 1) | (val_labels[:, 1] == 1))
    val_neg = np.sum((val_labels[:, 0] == 0) & (val_labels[:, 1] == 0))
    test_pos = np.sum((test_labels[:, 0] == 1) | (test_labels[:, 1] == 1))
    test_neg = np.sum((test_labels[:, 0] == 0) & (test_labels[:, 1] == 0))
    
    print(f"Train set: {len(train_indices)} samples")
    print(f"Pos/Neg: {train_pos}/{train_neg}")
    print(f"Positive ratio: {train_pos/len(train_indices):.3f}")
    
    print(f"\nValidation set: {len(val_indices)} samples")
    print(f"Pos/Neg: {val_pos}/{val_neg}")
    print(f"Positive ratio: {val_pos/len(val_indices):.3f}")
    
    print(f"\nTest set (age >= {age_threshold}): {len(test_indices)} samples")
    print(f"Pos/Neg: {test_pos}/{test_neg}")
    print(f"Positive ratio: {test_pos/len(test_indices):.3f}")

def plot_pr_curves(results_bce, results_kd, run_id, save_path):
    """Plot and save PR curves"""
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn')
    
    precision, recall, _ = precision_recall_curve(results_bce['labels'], 
                                                results_bce['predictions'])
    plt.plot(recall, precision, label=f'BCE (AUPRC={results_bce["auprc"]:.3f})', 
             linewidth=2, alpha=0.8)
    
    precision, recall, _ = precision_recall_curve(results_kd['labels'], 
                                                results_kd['predictions'])
    plt.plot(recall, precision, label=f'KD (AUPRC={results_kd["auprc"]:.3f})', 
             linewidth=2, alpha=0.8)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves (Run {run_id})', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.savefig(f'{save_path}/pr_curves_run_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def check_cuda():
    """Check CUDA availability and print device information"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

def check_age_distribution(info_df, train_indices, test_indices):
    """Check age distribution in train and test sets"""
    print("\nAge Distribution:")
    print(f"Train age range: {info_df.loc[train_indices, 'age'].min():.1f} - {info_df.loc[train_indices, 'age'].max():.1f}")
    print(f"Test age range: {info_df.loc[test_indices, 'age'].min():.1f} - {info_df.loc[test_indices, 'age'].max():.1f}")

def preprocess_data(data_dir, info_df, train_indices, n_components=1000, save_dir=None):
    """Preprocess data"""
    print("Loading training data...")
    train_data = []
    for sample_id in info_df['sample_id'].iloc[train_indices]:
        x = np.load(f"{data_dir}/{sample_id}.npy").astype(np.float32)
        train_data.append(x)
    train_data = np.array(train_data)
    
    print(f"Training data shape: {train_data.shape}")
    print("Fitting preprocessing models...")
    scaler = StandardScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    pca = PCA(n_components=n_components).fit(train_scaled)
    
    explained_var_ratio = pca.explained_variance_ratio_
    print(f"Total explained variance ratio: {np.sum(explained_var_ratio):.3f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Processing and saving all data...")
        
        for i, sample_id in enumerate(info_df['sample_id']):
            x = np.load(f"{data_dir}/{sample_id}.npy").astype(np.float32)
            x_scaled = scaler.transform(x.reshape(1, -1))
            x_reduced = pca.transform(x_scaled)
            np.save(f"{save_dir}/{sample_id}.npy", x_reduced.squeeze())
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(info_df)} samples")
        
        np.save(f"{save_dir}/pca_components.npy", pca.components_)
        np.save(f"{save_dir}/scaler_mean.npy", scaler.mean_)
        np.save(f"{save_dir}/scaler_scale.npy", scaler.scale_)
        
        print("Preprocessing completed!")
    
    return scaler, pca

def standardize_features(data_dir, info_df, train_indices, val_indices, test_indices, save_dir=None):
    """Standardize features"""
    def process_dataset(name, indices):
        print(f"\nProcessing {name} set...")
        data = []
        sample_ids = info_df['sample_id'].iloc[indices].values
        for sample_id in sample_ids:
            x = np.load(f"{data_dir}/{sample_id}.npy").astype(np.float32)
            data.append(x)
        data = np.array(data)  # shape: [n_samples, n_features]
        
        feature_means = np.mean(data, axis=0)  # shape: [n_features]
        feature_stds = np.std(data, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        data_scaled = (data - feature_means) / feature_stds
        print(f"Scaled data shape: {data_scaled.shape}")
        
        if save_dir:
            for i, (sample_id, x_scaled) in enumerate(zip(sample_ids, data_scaled)):
                np.save(f"{save_dir}/{sample_id}.npy", x_scaled)
                if (i + 1) % 1000 == 0:
                    print(f"Saved {i + 1}/{len(sample_ids)} samples")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    process_dataset('train', train_indices)
    process_dataset('validation', val_indices)
    process_dataset('test', test_indices)


# gradient clipping
class GradientQueue:
    def __init__(self, maxlen=10):
        self.queue = []
        self.maxlen = maxlen
        
    def add(self, value):
        if len(self.queue) >= self.maxlen:
            self.queue.pop(0)
        self.queue.append(value)
        
    def mean(self):
        if not self.queue:
            return 1.0
        return np.mean(self.queue)
    
    def std(self):
        if len(self.queue) <= 1:
            return 0.0
        return np.std(self.queue)

def gradient_clipping(model, gradnorm_queue):
    # allow 200% + 3 * std
    max_grad_norm = 2.0 * gradnorm_queue.mean() + 3 * gradnorm_queue.std()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.2f}, '
              f'allowed maximum value is {max_grad_norm:.2f}')
    return grad_norm

def get_model(model_type, d_input, n_snps, n_genes, args, device):
    if model_type == "MLP":
        model = MLP(d_input, d_hidden=args.d_hidden).to(device)
    elif model_type == "ctrMLP":
        model = ctrMLP(d_input, d_hidden=args.d_hidden, ema=args.ema, pos_tau=args.pos_tau, neg_tau=args.neg_tau, neg_rate=args.neg_rate, lamb=args.lamb).to(device)
    elif model_type == "AgeAwareMLP1":
        model = AgeAwareMLP1(d_input, d_hidden=args.d_hidden, use_consist=args.age1_use_consist, use_adversarial=args.age1_use_adversarial, use_cumulative_rate=args.use_cumulative_rate).to(device)
    elif model_type == "AgeAwareMLP2":
        model = AgeAwareMLP2(d_input, d_hidden=args.d_hidden, use_consist=args.age2_use_consist, use_ageloss=args.age2_use_ageloss, use_disentangle=args.age2_use_disentangle, use_cumulative_rate=args.use_cumulative_rate).to(device)
    elif model_type == "UGP_v1":
        model = UGP_v1(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)
    elif model_type == "UGP_v2":
        model = UGP_v2(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)
    elif model_type == "UGP_v3":
        model = UGP_v3(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout, n_gnn_layers=args.n_gnn_layers).to(device)
    elif model_type == "ctrUGP_v1":
        model = ctrUGP_v1(n_snps=n_snps, n_genes=n_genes, age_threshold=args.age_threshold, d_hidden=args.d_hidden, gene_dropout=args.gene_dropout, snp_dropout=args.snp_dropout, n_gnn_layers=args.n_gnn_layers, ema=args.ema, pos_tau=args.pos_tau, neg_tau=args.neg_tau, neg_rate=args.neg_rate, lamb=args.lamb).to(device)
    elif model_type == "AgeUGP_v1":
        model = AgeUGP_v1(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout, use_adversarial=args.age1_use_adversarial, use_consist=args.age1_use_consist, use_cumulative_rate=args.use_cumulative_rate).to(device)
    elif model_type == "AgeUGP_v2":
        model = AgeUGP_v2(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout, use_disentangle=args.age2_use_disentangle, use_ageloss=args.age2_use_ageloss, use_consist=args.age2_use_consist, use_cumulative_rate=args.use_cumulative_rate).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def get_optimizer(model, model_type, lr, use_adaptive_lr):
    if use_adaptive_lr:
        param_groups = []
        if model_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            if hasattr(model, 'age_layer'):
                param_groups.append({
                    'params': model.age_layer.parameters(),
                    'lr': lr * 0.1
                })
            if hasattr(model, 'age_predictor'):
                 param_groups.append({
                    'params': model.age_predictor.parameters(),
                    'lr': lr * 0.1
                })
        if model_type in ["MLP", "AgeAwareMLP1", "AgeAwareMLP2"]:
             if hasattr(model, 'model'):
                param_groups.append({
                    'params': model.model.parameters(),
                    'lr': lr
                })
        if model_type == "AgeAwareMLP2":
            if hasattr(model, 'main_head'):
                param_groups.append({
                    'params': model.main_head.parameters(),
                    'lr': lr
                })
        grouped_param_ids = set()
        for group in param_groups:
            for param in group['params']:
                grouped_param_ids.add(id(param))

        remaining_params = [p for p in model.parameters() if id(p) not in grouped_param_ids]
        if remaining_params:
             param_groups.append({
                'params': remaining_params,
                'lr': lr
            })
        
        param_groups = [pg for pg in param_groups if pg['params']]

        if not param_groups:
             print("Warning: No parameters passed to the optimizer. Using default settings.")
             optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(param_groups)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer

def mixup(inputs, labels, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_inputs, mixed_labels, lam
    
def asym_noise(labels, asym_noise_ratio): # labels: pd.DataFrame
    pos_indices = np.where(labels == 1)[0]
    n_pos = len(pos_indices)
    n_noise = int(n_pos * asym_noise_ratio)
    noise_indices = np.random.choice(pos_indices, size=n_noise, replace=False)
    labels.iloc[noise_indices] = 0
    return labels, noise_indices

# if __name__ == "__main__":
#     np.random.seed(22)
#     labels = pd.DataFrame({'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
#     labels, noise_indices = asym_noise(labels, 0.5)
#     print(labels)
#     print(noise_indices)