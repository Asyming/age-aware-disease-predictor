import sys
import os
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.teacher_models import MLP, AgeAwareMLP1, AgeAwareMLP2
# from src.student_models import SimpleLinear, XGBoost, LightGBM
from src.dataset import Dataset
from src.loss import KDLoss
from src.trainer import Trainer, KDTrainer
from src.utils import (
    set_random_seed, 
    seed_worker, 
    collate_fn, 
    split_by_age, 
    plot_pr_curves,
    plot_prediction_distribution,
    check_cuda
)

def parse_args():
    parser = argparse.ArgumentParser(description="Comparative study of BCE vs KD loss")
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=60)
    parser.add_argument("--data_dir", type=str, default='./data/ad',choices=['./data/ad','./data/ms','./data/uc'])
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--n_early_stop", type=int, default=5)
    # parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True)
    # parser.add_argument("--teacher_model_path", type=str, default=None)
    # parser.add_argument("--initialize_teacher_from_baseline", action='store_true')
    parser.add_argument("--teacher_type", type=str, default="AgeAwareMLP1", choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2"])
    parser.add_argument("--student_type", type=str, default="MLP", choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "SimpleLinear"])
    return parser.parse_args()

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[info_df['ancestry'] == args.ancestry].reset_index(drop=True)
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold)
    train_indices, val_indices, test_indices = splits_list[0]
    
    print("\nOriginal Data Distribution:")
    print(f"Train set (before noise):")
    print(f"Total: {len(train_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[train_indices, 'label'] == 1)}/{np.sum(info_df.loc[train_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[train_indices, 'label'] == 1):.3f}")
    
    train_pos_indices = train_indices[info_df.loc[train_indices, 'label'] == 1]
    n_noise = int(len(train_pos_indices) * args.noise_ratio)
    noise_indices = np.random.choice(train_pos_indices, n_noise, replace=False)
    info_df.loc[noise_indices, 'label'] = 0
    
    print(f"\nTrain set (after adding {args.noise_ratio*100:.1f}% noise to positive samples):")
    print(f"Total: {len(train_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[train_indices, 'label'] == 1)}/{np.sum(info_df.loc[train_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[train_indices, 'label'] == 1):.3f}")
    
    print(f"\nValidation set:")
    print(f"Total: {len(val_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[val_indices, 'label'] == 1)}/{np.sum(info_df.loc[val_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[val_indices, 'label'] == 1):.3f}")
    
    print(f"\nTest set (age >= {args.age_threshold}):")
    print(f"Total: {len(test_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[test_indices, 'label'] == 1)}/{np.sum(info_df.loc[test_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[test_indices, 'label'] == 1):.3f}")
    
    all_results = {
        'bce': {'auroc': [], 'auprc': []},
        'kd': {'auroc': [], 'auprc': []}
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f'experiments/{args.teacher_type}_{args.student_type}_{args.data_dir.split("/")[-1]}_age{args.age_threshold}_{args.noise_ratio}_es{args.n_early_stop}_{timestamp}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # Determine whether to create a log file and evaluate the training set based on whether there is noise
    has_noise = args.noise_ratio > 0
    log_file = None
    
    # Only create a log file when there is noise
    if has_noise:
        log_file = f'{exp_dir}/noise_predictions.log'
        with open(log_file, 'w') as f:
            f.write(f"Experiment Settings:\n")
            f.write(f"Noise ratio: {args.noise_ratio}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Alpha: {args.alpha}\n\n")
        
    for run in range(args.n_runs):
        if has_noise:
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Run {run+1}/{args.n_runs}\n")
                f.write(f"{'='*50}\n")
        
        train_dataset = Dataset(
            f"{args.data_dir}/X",
            info_df['sample_id'].iloc[train_indices].values,
            info_df['age'].iloc[train_indices].values,
            info_df['label'].iloc[train_indices].values,
            balanced_sampling=True
        )
        
        val_dataset = Dataset(
            f"{args.data_dir}/X",
            info_df['sample_id'].iloc[val_indices].values,
            info_df['age'].iloc[val_indices].values,
            info_df['label'].iloc[val_indices].values,
            balanced_sampling=False
        )
        
        test_dataset = Dataset(
            f"{args.data_dir}/X",
            info_df['sample_id'].iloc[test_indices].values,
            info_df['age'].iloc[test_indices].values,
            info_df['label'].iloc[test_indices].values,
            balanced_sampling=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=512,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # DataLoader for evaluating training set (no balanced sampling)
        train_eval_loader = None
        if has_noise:
            train_eval_dataset = Dataset(
                f"{args.data_dir}/X",
                info_df['sample_id'].iloc[train_indices].values,
                info_df['age'].iloc[train_indices].values,
                info_df['label'].iloc[train_indices].values,
                balanced_sampling=False
            )
            train_eval_loader = DataLoader(
                train_eval_dataset,
                batch_size=512,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                drop_last=False,
                collate_fn=collate_fn,
                pin_memory=True
            )
        
        d_input = np.load(f"{args.data_dir}/X/{info_df['sample_id'].iloc[0]}.npy").shape[0]
        
        # Teacher and Student structures and parameters
        if args.teacher_type == "MLP":
            teacher_model = MLP(d_input, d_hidden=64).to(device)
        elif args.teacher_type == "AgeAwareMLP1":
            teacher_model = AgeAwareMLP1(d_input, d_hidden=64).to(device)
        elif args.teacher_type == "AgeAwareMLP2":
            teacher_model = AgeAwareMLP2(d_input, d_hidden=64).to(device)
        
        if args.student_type == "MLP":
            student_model = MLP(d_input, d_hidden=64).to(device)
        elif args.student_type == "AgeAwareMLP1":
            student_model = AgeAwareMLP1(d_input, d_hidden=64).to(device)
        elif args.student_type == "AgeAwareMLP2":
            student_model = AgeAwareMLP2(d_input, d_hidden=64).to(device)
        # elif args.student_type == "SimpleLinear":
        #     student_model = SimpleLinear(d_input).to(device)
        

        teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.lr)
        student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
        
        teacher_criterion = nn.BCEWithLogitsLoss()
        student_criterion = KDLoss(temperature=args.temperature, alpha=args.alpha)
        
        teacher_trainer = Trainer(
            model=teacher_model, 
            criterion=teacher_criterion,
            optimizer=teacher_optimizer,
            device=device,
            model_name=f'teacher_run_{run}',
            save_dir=exp_dir,
            eval_interval=args.eval_interval,      # Evaluate every eval_interval steps
            n_steps=20000,                         # Total training steps
            n_early_stop=args.n_early_stop,        # Early stop if validation performance does not improve for n_early_stop times
            log_interval=args.log_interval         # Print loss every log_interval steps
        )
        teacher_results = teacher_trainer.train(train_loader, val_loader, test_loader, train_eval_loader if has_noise else None)
        teacher_model.eval()

        student_trainer = KDTrainer(
            model=student_model,
            teacher_model=teacher_model,
            criterion=student_criterion,
            optimizer=student_optimizer,
            device=device,
            model_name=f'student_run_{run}',
            teacher_model_path=os.path.join(exp_dir, f'teacher_run_{run}.pth'),
            save_dir=exp_dir,
            eval_interval=args.eval_interval,
            n_steps=20000,
            n_early_stop=args.n_early_stop,
            log_interval=args.log_interval
        )
        student_results = student_trainer.train(train_loader, val_loader, test_loader, train_eval_loader if has_noise else None)
        
        all_results['bce']['auroc'].append(teacher_results['test_metrics']['auroc'])
        all_results['bce']['auprc'].append(teacher_results['test_metrics']['auprc'])
        all_results['kd']['auroc'].append(student_results['test_metrics']['auroc'])
        all_results['kd']['auprc'].append(student_results['test_metrics']['auprc'])
        
        plot_pr_curves(teacher_results['test_metrics'], student_results['test_metrics'], run, exp_dir)
        
        if has_noise:
            with open(f'{exp_dir}/noise_predictions.log', 'a') as f:
                original_stdout = sys.stdout
                sys.stdout = f
                
                plot_prediction_distribution(
                    teacher_results=teacher_results,
                    student_results=student_results,
                    noise_indices=noise_indices,
                    train_indices=train_indices,
                    run_id=run,
                    save_path=exp_dir
                )
                
                sys.stdout = original_stdout

    summary = {
        'bce': {
            'auroc_mean': np.mean(all_results['bce']['auroc']),
            'auroc_std': np.std(all_results['bce']['auroc']),
            'auprc_mean': np.mean(all_results['bce']['auprc']),
            'auprc_std': np.std(all_results['bce']['auprc'])
        },
        'kd': {
            'auroc_mean': np.mean(all_results['kd']['auroc']),
            'auroc_std': np.std(all_results['kd']['auroc']),
            'auprc_mean': np.mean(all_results['kd']['auprc']),
            'auprc_std': np.std(all_results['kd']['auprc'])
        }
    }
    
    with open(f'{exp_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    check_cuda()
    set_random_seed()
    summary = run_experiment(args) 