import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from src.teacher_models import AgeAwareMLP1, AgeAwareMLP2, AgeUGP_v1, AgeUGP_v2
from src.dataset import Dataset
from src.utils import set_random_seed, split_by_age, seed_worker, collate_fn
from scripts.train4 import parse_args

def load_model_and_data(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[info_df['ancestry'] == args.ancestry].reset_index(drop=True)
    
    set_random_seed()
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold)
    train_indices, _, _ = splits_list[0]
    
    dataset = Dataset(
        f"{args.data_dir}/X",
        info_df['sample_id'].iloc[train_indices].values,
        info_df['age'].iloc[train_indices].values,
        info_df['label'].iloc[train_indices].values,
        balanced_sampling=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    d_input = dataset[0]['feat'].shape[0]
    if model_type == 'AgeAwareMLP1':
        model = AgeAwareMLP1(d_input, d_hidden=64)
    elif model_type == 'AgeAwareMLP2':
        model = AgeAwareMLP2(d_input, d_hidden=64)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_path = os.path.join(exp_dir, 'teacher_run_1.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, 'student_run_1.pth')
    
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, dataloader, device

def analyze_transition_matrix(model, dataloader, save_dir, exp_name, device):
    """分析转移矩阵"""
    if isinstance(model, AgeAwareMLP1):
        results = {'ages': [], 'matrices': []}
        all_ages = torch.arange(40, 71, dtype=torch.float32, device=device)
        all_ages_normed = (all_ages - 40) / (70 - 40)
        
        with torch.no_grad():
            for age, age_normed in zip(all_ages, all_ages_normed):
                trans_matrix = model.get_transition_matrix(age_normed.unsqueeze(0))
                results['ages'].append(age.item())
                results['matrices'].append(trans_matrix.cpu().squeeze().numpy().tolist())
        
        plt.figure(figsize=(15, 10))
        ages_np = np.array(results['ages'])
        matrices_np = np.array(results['matrices'])
        
        transitions = [
            (0, 0, '0→0', 'P(negative→negative)'),
            (0, 1, '0→1', 'P(negative→positive)'),
            (1, 0, '1→0', 'P(positive→negative)'),
            (1, 1, '1→1', 'P(positive→positive)')
        ]
        
        for i, (row, col, label, title) in enumerate(transitions):
            plt.subplot(2, 2, i+1)
            plt.plot(ages_np, matrices_np[:, row, col], marker='o', 
                    linestyle='-', markersize=4, alpha=0.5, linewidth=1.5)
            plt.title(title)
            plt.xlabel('Age')
            plt.ylabel(f'Probability {label}')
            plt.grid(alpha=0.3)
            plt.ylim(-0.05, 1.05)
        
        plt.suptitle(exp_name, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'transition_matrix_analysis.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        output = {str(age): matrix for age, matrix in zip(results['ages'], results['matrices'])}
        
    elif isinstance(model, AgeAwareMLP2):
        results = {'ages': [], 'matrices': []}
        
        with torch.no_grad():
            for feats, _, ages in dataloader:
                feats = feats.to(device)
                ages = ages.to(device)
                age_norm = (ages - 40) / (70 - 40)
                _, age_feat = model.get_intermediate_features(feats)
                trans_matrix = model.get_transition_matrix(age_norm, age_feat)
                matrices = trans_matrix.cpu().squeeze().numpy()
                ages = ages.cpu().squeeze().numpy()
                
                results['ages'].extend(ages.tolist())
                results['matrices'].extend(matrices.tolist())
        
        age_stats = {}
        for age, matrix in zip(results['ages'], results['matrices']):
            age = int(age)
            if age not in age_stats:
                age_stats[age] = {'P00': [], 'P01': [], 'P10': [], 'P11': []}
            age_stats[age]['P00'].append(matrix[0][0])
            age_stats[age]['P01'].append(matrix[0][1])
            age_stats[age]['P10'].append(matrix[1][0])
            age_stats[age]['P11'].append(matrix[1][1])
        
        stats = {}
        for age in range(40, 71):
            stats[age] = {
                'P00': {'mean': np.nan, 'std': np.nan},
                'P01': {'mean': np.nan, 'std': np.nan},
                'P10': {'mean': np.nan, 'std': np.nan},
                'P11': {'mean': np.nan, 'std': np.nan}
            }
            if age in age_stats:
                for key in ['P00', 'P01', 'P10', 'P11']:
                    data = age_stats[age][key]
                    if data:
                        stats[age][key]['mean'] = float(np.mean(data))
                        stats[age][key]['std'] = float(np.std(data)) if len(data) > 1 else 0.0
        
        plt.figure(figsize=(15, 10))
        transitions = [
            ('P00', '0→0', 'P(negative→negative)'),
            ('P01', '0→1', 'P(negative→positive)'),
            ('P10', '1→0', 'P(positive→negative)'),
            ('P11', '1→1', 'P(positive→positive)')
        ]
        
        for i, (prob_key, label, title) in enumerate(transitions):
            plt.subplot(2, 2, i+1)
            ages = np.arange(40, 71)
            means = [stats[age][prob_key]['mean'] for age in ages]
            stds = [stats[age][prob_key]['std'] for age in ages]
            valid = ~np.isnan(means)
            
            plt.errorbar(np.array(ages)[valid], np.array(means)[valid], 
                        yerr=np.array(stds)[valid], fmt='-o', markersize=4,
                        linewidth=1.5, ecolor='gray', capsize=3)
            plt.title(title)
            plt.xlabel('Age')
            plt.ylabel('Probability')
            plt.grid(alpha=0.3)
            plt.ylim(-0.05, 1.05)
        
        plt.suptitle(f"{exp_name} (Age Range: 40-70)", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'transition_matrix_stats.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        output = {str(k): v for k, v in stats.items()}
        
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    return output

def main():
    diseases = ['ad']#, 'ms', 'uc']
    age_thresholds = {'ad': 65}#, 'ms': 65, 'uc': 65}
    ancestry = 'EUR'
    
    configs = [
        {'mode': 'teacher', 'model_type': 'AgeAwareMLP1'},
        {'mode': 'student', 'model_type': 'AgeAwareMLP1', 'student_type': 'AgeAwareMLP1'},
        {'mode': 'teacher', 'model_type': 'AgeAwareMLP2'},
        {'mode': 'student', 'model_type': 'AgeAwareMLP2', 'student_type': 'AgeAwareMLP2'}
    ]
    
    all_transitions = {}
    
    for disease in diseases:
        data_dir = f'./data/{disease}'
        age_threshold = age_thresholds[disease]
        
        for config in configs:
            if config['mode'] == 'teacher':
                exp_pattern = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}_run{run}'
            else:
                exp_pattern = f"{disease}_{config['mode']}_{config['model_type']}_{config['student_type']}_age{age_threshold}_es20_run1"
            
            exp_dir = os.path.join('experiments', exp_pattern)
            
            if not os.path.exists(exp_dir):
                print(f"Skipping {exp_pattern} - directory not found")
                continue
            
            print(f"\nAnalyzing {exp_pattern}")
            model, dataloader, device = load_model_and_data(
                exp_dir, config['model_type'], data_dir, ancestry, age_threshold
            )
            
            transitions = analyze_transition_matrix(model, dataloader, exp_dir, exp_pattern, device)
            all_transitions[exp_pattern] = transitions

    with open('age_transition_entry.json', 'w') as f:
        json.dump(all_transitions, f, indent=4, ensure_ascii=False, default=lambda o: o.__dict__)

if __name__ == '__main__':
    main()