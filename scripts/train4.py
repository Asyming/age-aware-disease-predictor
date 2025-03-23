import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/ageaware"))
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import dgl
import pickle
import json
from src.teacher_models import MLP, AgeAwareMLP1, AgeAwareMLP2, UGP_v1, UGP_v2
from src.dataset import Dataset
from src.loss import KDLoss
from src.trainer import Trainer, Trainer_g, KDTrainer
from src.utils import (set_random_seed, seed_worker, collate_fn, split_by_age, check_cuda)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher or Student Models")
    # experiment settings
    parser.add_argument("--exp_name", type=str, default='test1')
    parser.add_argument("--teacher_model_exp_name", type=str, default=None)
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=65)
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc','./data/ad_new'])
    parser.add_argument("--age_range", type=str, default='40-70', help='min-max')
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument("--teacher_type", type=str, default="MLP", choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "UGP_v1", "UGP_v2"])
    parser.add_argument("--student_type", type=str, default=None, choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2", "UGP_v1", "UGP_v2"])
    # model settings
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--n_early_stop", type=int, default=20)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--teacher_model_lr", type=float, default=1e-4)
    parser.add_argument("--teacher_model_es", type=int, default=20)
    parser.add_argument("--teacher_model_eval", type=int, default=50)
    parser.add_argument("--snp_dropout", type=float, default=0.99)
    parser.add_argument("--gene_dropout", type=float, default=0.5)
    # ablation study
    parser.add_argument("--age1_use_consist", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age1_use_adversarial", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_consist", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_ageloss", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_disentangle", type=lambda x: x.lower() == 'true', default=True)
    # auxiliary tools
    parser.add_argument("--use_adptive_lr", action='store_true')
    parser.add_argument("--use_cumulative_rate", action='store_true')
    return parser.parse_args()

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[(info_df['ancestry'] == args.ancestry) & (info_df['age'] >= int(args.age_range.split('-')[0])) & (info_df['age'] <= int(args.age_range.split('-')[1]))].reset_index(drop=True)
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold, n_splits=args.n_runs)
    snp_ids, batched_g = None, None
    if args.teacher_type in ["UGP_v1", "UGP_v2"] or args.student_type in ["UGP_v1", "UGP_v2"]:
        with open(f"{args.data_dir}/gene2snps.pkl", "rb") as f:
            gene2snps = pickle.load(f)
        n_genes = len(gene2snps.keys())
        snp_ids = []
        for gene in gene2snps.keys():
            snp_ids += gene2snps[gene]
        n_snps = len(np.unique(snp_ids))
        print('n_genes:', n_genes, 'n_snps:', n_snps, 'n_nodes:', len(snp_ids))
        graph_list = []
        for snps in gene2snps.values():
            graph_list.append(dgl.graph(data=[], num_nodes=len(snps)))
        batched_g = dgl.batch(graph_list)
    all_results = {
        'teacher': {'auroc': [], 'auprc': []},
        'student': {'auroc': [], 'auprc': []}
    }
    for run in range(1, args.n_runs + 1):
        # debug info
        print(f"\nRun {run}/{args.n_runs}")
        train_indices, val_indices, test_indices = splits_list[run-1]
        print(f"\nSplit {run} Data Distribution:")
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
        
        if args.mode == "teacher":
            exp_dir = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}'
        elif args.mode == "student":
            exp_dir = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_{args.student_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}'
        os.makedirs(exp_dir, exist_ok=True)

        train_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[train_indices].values, ages=info_df['age'].iloc[train_indices].values, labels=info_df['label'].iloc[train_indices].values, balanced_sampling=True)
        val_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[val_indices].values, ages=info_df['age'].iloc[val_indices].values, labels=info_df['label'].iloc[val_indices].values, balanced_sampling=False)
        test_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[test_indices].values, ages=info_df['age'].iloc[test_indices].values, labels=info_df['label'].iloc[test_indices].values, balanced_sampling=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=True, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn, pin_memory=True)
        
        d_input = np.load(f"{args.data_dir}/X/{info_df['sample_id'].iloc[0]}.npy").shape[0]
        if args.mode == "teacher":
            if args.teacher_type == "MLP":
                teacher_model = MLP(d_input, d_hidden=args.d_hidden).to(device)
            elif args.teacher_type == "AgeAwareMLP1":
                teacher_model = AgeAwareMLP1(d_input, d_hidden=args.d_hidden, use_consist=args.age1_use_consist, use_adversarial=args.age1_use_adversarial, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.teacher_type == "AgeAwareMLP2":
                teacher_model = AgeAwareMLP2(d_input, d_hidden=args.d_hidden, use_consist=args.age2_use_consist, use_ageloss=args.age2_use_ageloss, use_disentangle=args.age2_use_disentangle, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.teacher_type == "UGP_v1":
                teacher_model = UGP_v1(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)
            elif args.teacher_type == "UGP_v2":
                teacher_model = UGP_v2(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)

            if args.use_adptive_lr:
                param_groups = []
                if args.teacher_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': teacher_model.age_layer.parameters(),
                        'lr': args.lr * 0.1
                    })
                if args.teacher_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': teacher_model.age_predictor.parameters(),
                        'lr': args.lr * 0.1
                    })
                if args.teacher_type in ["MLP", "AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': teacher_model.model.parameters(),
                        'lr': args.lr
                    })
                if args.teacher_type == "AgeAwareMLP2":
                    param_groups.append({
                        'params': teacher_model.main_head.parameters(),
                        'lr': args.lr
                    })
                optimizer = torch.optim.AdamW(param_groups)
            else: 
                optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()
            if args.teacher_type in ["UGP_v1", "UGP_v2"]:
                trainer = Trainer_g(
                    model=teacher_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    model_name=f'teacher_run_{run}',
                    save_dir=exp_dir,
                    snp_ids=torch.LongTensor(snp_ids).to(device),
                    batched_g=batched_g.to(device),
                    eval_interval=args.eval_interval,
                    n_steps=args.n_steps,
                    n_early_stop=args.n_early_stop,
                    log_interval=args.log_interval
                )
                teacher_results = trainer.train(train_loader, val_loader, test_loader)
                all_results['teacher']['auprc'].append(teacher_results['test_metrics']['auprc'])
            else:
                trainer = Trainer(
                    model=teacher_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    model_name=f'teacher_run_{run}',
                    save_dir=exp_dir,
                    eval_interval=args.eval_interval,
                    n_steps=args.n_steps,
                    n_early_stop=args.n_early_stop,
                    log_interval=args.log_interval
                )
                teacher_results = trainer.train(train_loader, val_loader, test_loader)
                all_results['teacher']['auprc'].append(teacher_results['test_metrics']['auprc'])

        elif args.mode == "student": 
            if args.teacher_type == "MLP":
                teacher_model = MLP(d_input, d_hidden=args.d_hidden).to(device)
            elif args.teacher_type == "AgeAwareMLP1":
                teacher_model = AgeAwareMLP1(d_input, d_hidden=args.d_hidden, use_consist=args.age1_use_consist, use_adversarial=args.age1_use_adversarial, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.teacher_type == "AgeAwareMLP2":
                teacher_model = AgeAwareMLP2(d_input, d_hidden=args.d_hidden, use_consist=args.age2_use_consist, use_ageloss=args.age2_use_ageloss, use_disentangle=args.age2_use_disentangle, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.teacher_type == "UGP_v1":
                teacher_model = UGP_v1(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)
            elif args.teacher_type == "UGP_v2":
                teacher_model = UGP_v2(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)

            if args.student_type == "MLP":
                student_model = MLP(d_input, d_hidden=args.d_hidden).to(device)
            elif args.student_type == "AgeAwareMLP1":
                student_model = AgeAwareMLP1(d_input, d_hidden=args.d_hidden, use_consist=args.age1_use_consist, use_adversarial=args.age1_use_adversarial, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.student_type == "AgeAwareMLP2":
                student_model = AgeAwareMLP2(d_input, d_hidden=args.d_hidden, use_consist=args.age2_use_consist, use_ageloss=args.age2_use_ageloss, use_disentangle=args.age2_use_disentangle, use_cumulative_rate=args.use_cumulative_rate).to(device)
            elif args.student_type == "UGP_v1":
                student_model = UGP_v1(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)
            elif args.student_type == "UGP_v2":
                student_model = UGP_v2(n_snps=n_snps, n_genes=n_genes, snp_dropout=args.snp_dropout, gene_dropout=args.gene_dropout).to(device)

            if args.use_adptive_lr:
                param_groups = []
                if args.student_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': student_model.age_layer.parameters(),
                        'lr': args.lr * 0.1
                    })
                if args.student_type in ["AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': student_model.age_predictor.parameters(),
                        'lr': args.lr * 0.1
                    })
                if args.student_type in ["MLP", "AgeAwareMLP1", "AgeAwareMLP2"]:
                    param_groups.append({
                        'params': student_model.model.parameters(),
                        'lr': args.lr
                    })
                if args.student_type == "AgeAwareMLP2":
                    param_groups.append({
                        'params': student_model.main_head.parameters(),
                        'lr': args.lr
                    })
                optimizer = torch.optim.AdamW(param_groups)
            else:
                optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
            criterion = KDLoss(temperature=args.temperature, alpha=args.alpha)
            kd_trainer = KDTrainer(
                model=student_model,
                teacher_model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=f'student_run_{run}',
                teacher_model_path=f'experiments/{args.teacher_model_exp_name if args.teacher_model_exp_name else args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_teacher_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.teacher_model_es}_eval{args.teacher_model_eval}_lr{args.teacher_model_lr}/teacher_run_{run}.pth',
                save_dir=exp_dir,
                snp_ids=torch.LongTensor(snp_ids).to(device) if snp_ids else None,
                batched_g=batched_g.to(device) if batched_g else None,
                eval_interval=args.eval_interval,
                n_steps=args.n_steps,
                n_early_stop=args.n_early_stop,
                log_interval=args.log_interval
            )
            student_results = kd_trainer.train(train_loader, val_loader, test_loader)
            all_results['student']['auprc'].append(student_results['test_metrics']['auprc'])

    summary = {
        args.mode: {
            'auprc_mean': np.mean(all_results[args.mode]['auprc']),
            'auprc_std': np.std(all_results[args.mode]['auprc'])
        }
    }
    with open(f'{exp_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nTraining completed. Summary saved.")
    return summary

if __name__ == '__main__':
    args = parse_args()
    check_cuda()
    set_random_seed()
    run_experiment(args)