import os
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import json

from src.teacher_models import MLP, AgeAwareMLP1, AgeAwareMLP2, AgeAwareMLP3
from src.dataset import Dataset
from src.loss import KDLoss
from src.trainer import Trainer, KDTrainer
from src.utils import (
    set_random_seed, 
    seed_worker, 
    collate_fn, 
    split_by_age,
    check_cuda,
    preprocess_data,
    standardize_features
)
#
def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher or Student Models")
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=65)
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc'])
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--splitid", type=int, default=2)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument("--n_early_stop", type=int, default=20)
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument("--teacher_model_path", type=str, default=None)
    #parser.add_argument("--initialize_teacher_from_baseline", action='store_true', help="If need to initialize teacher from baseline weights")
    parser.add_argument("--teacher_type", type=str, default="MLP", choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "AgeAwareMLP3"])
    parser.add_argument("--student_type", type=str, default=None, choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "AgeAwareMLP3"])
    return parser.parse_args()

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[info_df['ancestry'] == args.ancestry].reset_index(drop=True)
    splits_list = split_by_age(
        info_df['label'].values, info_df['age'].values, args.age_threshold
    )
    train_indices, val_indices, test_indices = splits_list[args.splitid]

    print("\nData Distribution:")
    print(f"Train set: {len(train_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[train_indices, 'label'] == 1)}/"
          f"{np.sum(info_df.loc[train_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[train_indices, 'label'] == 1):.3f}")
    
    print(f"\nValidation set: {len(val_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[val_indices, 'label'] == 1)}/"
          f"{np.sum(info_df.loc[val_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[val_indices, 'label'] == 1):.3f}")
    
    print(f"\nTest set (age >= {args.age_threshold}): {len(test_indices)} samples")
    print(f"Pos/Neg: {np.sum(info_df.loc[test_indices, 'label'] == 1)}/"
          f"{np.sum(info_df.loc[test_indices, 'label'] == 0)}")
    print(f"Positive ratio: {np.mean(info_df.loc[test_indices, 'label'] == 1):.3f}")
    
    all_results = {
        'teacher': {'auroc': [], 'auprc': []},
        'student': {'auroc': [], 'auprc': []}
    }
    
    # Preprocess data
    # processed_data_dir = os.path.join(args.data_dir, 'processed_feature_scaled')
    # if not os.path.exists(processed_data_dir):
    #     print("\nStandardizing features within each dataset...")
    #     standardize_features(
    #         data_dir=f"{args.data_dir}/X",
    #         info_df=info_df,
    #         train_indices=train_indices,
    #         val_indices=val_indices,
    #         test_indices=test_indices,
    #         save_dir=processed_data_dir
    #     )
    
    for run in range(1, args.n_runs + 1):
        print(f"\nRun {run}/{args.n_runs}")
        
        if args.mode == "teacher":
            exp_dir = f'experiments/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_age{args.age_threshold}_es{args.n_early_stop}_run{run}'
        elif args.mode == "student":
            exp_dir = f'experiments/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_{args.student_type}_age{args.age_threshold}_es{args.n_early_stop}_run{run}'
        os.makedirs(exp_dir, exist_ok=True)

        train_dataset = Dataset(
             feat_path=f"{args.data_dir}/X",
            sample_ids=info_df['sample_id'].iloc[train_indices].values,
            ages=info_df['age'].iloc[train_indices].values,
            labels=info_df['label'].iloc[train_indices].values,
            balanced_sampling=True
        )
        
        val_dataset = Dataset(
            feat_path=f"{args.data_dir}/X",
            sample_ids=info_df['sample_id'].iloc[val_indices].values,
            ages=info_df['age'].iloc[val_indices].values,
            labels=info_df['label'].iloc[val_indices].values,
            balanced_sampling=False
        )
        
        test_dataset = Dataset(
            feat_path=f"{args.data_dir}/X",
            sample_ids=info_df['sample_id'].iloc[test_indices].values,
            ages=info_df['age'].iloc[test_indices].values,
            labels=info_df['label'].iloc[test_indices].values,
            balanced_sampling=False
        )
    
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=128,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        d_input = np.load(f"{args.data_dir}/X/{info_df['sample_id'].iloc[0]}.npy").shape[0]
        
        if args.mode == "teacher":
            if args.teacher_type == "MLP":
                teacher_model = MLP(d_input, d_hidden=64).to(device)
            elif args.teacher_type == "AgeAwareMLP1":
                teacher_model = AgeAwareMLP1(d_input, d_hidden=64).to(device)
            elif args.teacher_type == "AgeAwareMLP2":
                teacher_model = AgeAwareMLP2(d_input, d_hidden=64).to(device)
            elif args.teacher_type == "AgeAwareMLP3":
                teacher_model = AgeAwareMLP3(d_input, d_hidden=64).to(device)
            
            # if args.initialize_teacher_from_baseline and args.teacher_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            #     baseline_path = "./experiments/teacher_MLP_age60_run1/teacher_run_1.pth"  # Adjust path as needed
            #     if os.path.exists(baseline_path):
            #         teacher_model.load_baseline_weights(baseline_path)
            #         print(f"Loaded baseline weights from {baseline_path}")

            # param_groups = []

            # if args.teacher_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            #     param_groups.append({
            #         'params': teacher_model.age_layer.parameters(),
            #         'lr': args.lr * 0.1
            #     })

            # if args.teacher_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            #     param_groups.append({
            #         'params': teacher_model.age_predictor.parameters(),
            #         'lr': args.lr * 0.1
            #     })

            # if args.teacher_type in ["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "AgeAwareMLP3"]:
            #     param_groups.append({
            #         'params': teacher_model.model.parameters(),
            #         'lr': args.lr
            #     })

            # if args.teacher_type == "AgeAwareMLP2":
            #     param_groups.append({
            #         'params': teacher_model.main_head.parameters(),
            #         'lr': args.lr
            #     })
            # optimizer = torch.optim.AdamW(param_groups)
            optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()
            
            trainer = Trainer(
                model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=f'teacher_run_{run}',
                save_dir=exp_dir,
                eval_interval=args.eval_interval,
                n_steps=20000,
                n_early_stop=args.n_early_stop,
                log_interval=args.log_interval
            )
            
            teacher_results = trainer.train(train_loader, val_loader, test_loader)
            teacher_model.eval()
            
            all_results['teacher']['auprc'].append(teacher_results['test_metrics']['auprc'])
        
        elif args.mode == "student": 
            teacher_type = args.teacher_type
            if teacher_type == "MLP":
                teacher_model = MLP(d_input, d_hidden=64).to(device)
            elif teacher_type == "AgeAwareMLP1":
                teacher_model = AgeAwareMLP1(d_input, d_hidden=64).to(device)
            elif teacher_type == "AgeAwareMLP2":
                teacher_model = AgeAwareMLP2(d_input, d_hidden=64).to(device)
            elif teacher_type == "AgeAwareMLP3":
                teacher_model = AgeAwareMLP3(d_input, d_hidden=64).to(device)
            teacher_model.eval()
            
            if args.student_type == "MLP":
                student_model = MLP(d_input, d_hidden=64).to(device)
            elif args.student_type == "AgeAwareMLP1":
                student_model = AgeAwareMLP1(d_input, d_hidden=64).to(device)
            elif args.student_type == "AgeAwareMLP2":
                student_model = AgeAwareMLP2(d_input, d_hidden=64).to(device)
            elif args.student_type == "AgeAwareMLP3":
                student_model = AgeAwareMLP3(d_input, d_hidden=64).to(device)


            # param_groups = []

            # if args.student_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            #     param_groups.append({
            #         'params': student_model.age_layer.parameters(),
            #         'lr': args.lr * 0.1
            #     })

            # if args.student_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
            #     param_groups.append({
            #         'params': student_model.age_predictor.parameters(),
            #         'lr': args.lr * 0.1
            #     })

            # if args.student_type in ["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "AgeAwareMLP3"]:
            #     param_groups.append({
            #         'params': student_model.model.parameters(),
            #         'lr': args.lr
            #     })

            # if args.student_type == "AgeAwareMLP2":
            #     param_groups.append({
            #         'params': student_model.main_head.parameters(),
            #         'lr': args.lr
            #     })
            # optimizer = torch.optim.AdamW(param_groups)
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
            criterion = KDLoss(temperature=args.temperature, alpha=args.alpha)
            
            kd_trainer = KDTrainer(
                model=student_model,
                teacher_model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=f'student_run_{run}',
                teacher_model_path=args.teacher_model_path,
                save_dir=exp_dir,
                eval_interval=args.eval_interval,
                n_steps=20000,
                n_early_stop=args.n_early_stop,
                log_interval=args.log_interval
            )
            
            student_results = kd_trainer.train(train_loader, val_loader, test_loader)
            student_model.eval()
        
            all_results['student']['auprc'].append(student_results['test_metrics']['auprc'])
    
    summary = {
        'teacher': {
            #'auroc_mean': np.mean(all_results['teacher']['auroc']) if all_results['teacher']['auroc'] else None,
            #'auroc_std': np.std(all_results['teacher']['auroc']) if all_results['teacher']['auroc'] else None,
            'auprc_mean': np.mean(all_results['teacher']['auprc']) if all_results['teacher']['auprc'] else None,
            'auprc_std': np.std(all_results['teacher']['auprc']) if all_results['teacher']['auprc'] else None
        },
        'student': {
            #'auroc_mean': np.mean(all_results['student']['auroc']) if all_results['student']['auroc'] else None,
            #'auroc_std': np.std(all_results['student']['auroc']) if all_results['student']['auroc'] else None,
            'auprc_mean': np.mean(all_results['student']['auprc']) if all_results['student']['auprc'] else None,
            'auprc_std': np.std(all_results['student']['auprc']) if all_results['student']['auprc'] else None
        }
    }
    
    with open(f'{exp_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\nTraining completed. Summary saved.")
    return summary


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    check_cuda()
    set_random_seed()
    run_experiment(args)