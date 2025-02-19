import numpy as np
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
def split_by_age(labels, ages, age_threshold, n_splits=6):
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

def plot_prediction_distribution(teacher_results, student_results, noise_indices, train_indices, run_id, save_path, filename=None):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    train_logits = teacher_results['train_metrics']['predictions'].squeeze()
    train_probs = 1 / (1 + np.exp(-train_logits))
    noise_mask = np.isin(train_indices, noise_indices)

    sorted_indices = np.argsort(train_probs)[::-1]
    sorted_probs = train_probs[sorted_indices]
    sorted_mask = noise_mask[sorted_indices]
    
    x_coords = np.arange(len(sorted_probs))
    plt.vlines(x_coords[~sorted_mask], ymin=0, ymax=sorted_probs[~sorted_mask], 
              colors='lightgray', alpha=0.8, linewidth=0.1)
    plt.vlines(x_coords[sorted_mask], ymin=0, ymax=sorted_probs[sorted_mask], 
              colors='red', alpha=0.8, linewidth=0.1)
    
    threshold_idx = np.searchsorted(sorted_probs[::-1], 0.5)
    threshold_idx = len(sorted_probs) - threshold_idx

    plt.axvline(x=threshold_idx, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Teacher Model Training Set Predictions')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Prediction Probability')
    plt.legend(['Normal samples', 'Label-noise samples'])
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.subplot(1, 2, 2)
    train_logits = student_results['train_metrics']['predictions'].squeeze()
    train_probs = 1 / (1 + np.exp(-train_logits))
    
    sorted_indices = np.argsort(train_probs)[::-1]
    sorted_probs = train_probs[sorted_indices]
    sorted_mask = noise_mask[sorted_indices]
    
    x_coords = np.arange(len(sorted_probs))
    plt.vlines(x_coords[~sorted_mask], ymin=0, ymax=sorted_probs[~sorted_mask], 
              colors='lightgray', alpha=0.8, linewidth=0.1)
    plt.vlines(x_coords[sorted_mask], ymin=0, ymax=sorted_probs[sorted_mask], 
              colors='red', alpha=0.8, linewidth=0.1)
    
    threshold_idx = np.searchsorted(sorted_probs[::-1], 0.5)
    threshold_idx = len(sorted_probs) - threshold_idx
    plt.axvline(x=threshold_idx, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Student Model Training Set Predictions')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Prediction Probability')
    plt.legend(['Normal samples', 'Label-noise samples'])
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if filename is None:
        filename = f'pred_dist_run_{run_id}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTraining Set Distribution Statistics:")
    for model in ['Teacher', 'Student']:
        logits = (teacher_results['train_metrics']['predictions'] if model == 'Teacher' 
                 else student_results['train_metrics']['predictions'])
        probs = 1 / (1 + np.exp(-logits.squeeze()))
        
        print(f"\n{model} Model:")
        print(f"Noise samples (original pos, labeled as neg):")
        print(f"Mean prob: {probs[noise_mask].mean():.3f}")
        print(f"Samples with prob > 0.5: {np.sum((probs > 0.5) & noise_mask)}/{np.sum(noise_mask)}")

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
