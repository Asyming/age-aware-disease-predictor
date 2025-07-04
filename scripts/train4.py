import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/zisen/ageaware"))
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import dgl
# dgl.seed(22)
# # os.environ['DGLBACKEND'] = 'pytorch'
# # os.environ['DGL_ENABLE_GRAPH_SERIALIZATION'] = '0'
import pickle
import json

from src.args import parse_train_args
from src.dataset import Dataset
from src.loss import KDLoss, BGCELoss
from src.trainer import Trainer, KDTrainer
from src.utils import set_random_seed, seed_worker, collate_fn, split_by_age, check_data, check_cuda, get_model, get_optimizer, asym_noise

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    if args.data_dir == "./data/mix_1":
        info_df = pd.read_csv(f"{args.data_dir}/sample_info_{args.mix_target}.csv")
    info_df = info_df[(info_df['ancestry'] == args.ancestry) & (info_df['age'] >= int(args.age_range.split('-')[0])) & (info_df['age'] <= int(args.age_range.split('-')[1]))].reset_index(drop=True)
    d_input = np.load(f"{args.data_dir}/X/{info_df['sample_id'].iloc[0]}.npy").shape[0]
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold, n_splits=args.n_runs)
    snp_ids, batched_g, gene_g = None, None, None
    n_snps, n_genes = 0, 0
    if args.teacher_type in ["UGP_v1", "UGP_v2", "AgeUGP_v1", "AgeUGP_v2", "UGP_v3", "ctrUGP_v1"] or args.student_type in ["UGP_v1", "UGP_v2", "AgeUGP_v1", "AgeUGP_v2", "UGP_v3", "ctrUGP_v1"]:
        with open(f"{args.data_dir}/gene2snps.pkl", "rb") as f:
            gene2snps = pickle.load(f)
        n_genes = len(gene2snps.keys())
        snp_ids = []
        for gene in gene2snps.keys():
            snp_ids += gene2snps[gene]
        n_snps = len(np.unique(snp_ids))
        print('n_genes:', n_genes, 'n_snps:', n_snps, 'n_nodes:', len(snp_ids))
        graph_list = []
        for snps in gene2snps.values(): # snps: list of snp ids for each gene.
            graph_list.append(dgl.graph(data=[], num_nodes=len(snps)))
        batched_g = dgl.batch(graph_list)
        if args.teacher_type in ["UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] or args.student_type in ["UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"]:
            gene_g = dgl.load_graphs(f"{args.data_dir}/ggi_subgraph.bin")[0][0].to(device)

    all_results = {
        'teacher': {'auroc': [], 'auprc': [], 'val_auprc': [], 'overfiting_rate': []},
        'student': {'auroc': [], 'auprc': [], 'val_auprc': [], 'overfiting_rate': []},
        'remark': args.remark
    }

    for run in range(1, args.n_runs + 1):
        print(f"\nRun {run}/{args.n_runs}")
        labels = info_df['label'].copy()
        train_indices, val_indices, test_indices = splits_list[run-1]
        if args.use_asym_noise and args.asym_noise_ratio > 0:
            labels.iloc[train_indices] , _= asym_noise(labels.iloc[train_indices], args.asym_noise_ratio)
        check_data(info_df, train_indices, val_indices, test_indices, run_id = run, age_threshold = args.age_threshold)

        if args.mode == "teacher":
            exp_dir = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}_use_bgce{args.use_bgce}_use_lc{args.use_label_correction}_use_mixup{args.use_mixup}'
        elif args.mode == "student":
            exp_dir = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_{args.student_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}_use_bgce{args.use_bgce}_use_lc{args.use_label_correction}_use_mixup{args.use_mixup}'
        os.makedirs(exp_dir, exist_ok=True)

        train_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[train_indices].values, ages=info_df['age'].iloc[train_indices].values, labels=labels.iloc[train_indices].values, balanced_sampling=True)
        val_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[val_indices].values, ages=info_df['age'].iloc[val_indices].values, labels=labels.iloc[val_indices].values, balanced_sampling=False)
        test_dataset = Dataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[test_indices].values, ages=info_df['age'].iloc[test_indices].values, labels=labels.iloc[test_indices].values, balanced_sampling=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=True, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn, pin_memory=True)

        if args.mode == "teacher":
            teacher_model = get_model(args.teacher_type, d_input, n_snps, n_genes, args, device)
            optimizer = get_optimizer(teacher_model, args.teacher_type, args.lr, args.use_adaptive_lr)
            if args.use_bgce:
                criterion = BGCELoss(q=args.bgce_q)
            else:
                criterion = nn.BCEWithLogitsLoss()
            trainer = Trainer(
                model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=f'teacher_run_{run}',
                save_dir=exp_dir,
                snp_ids=torch.LongTensor(snp_ids) if args.teacher_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                batched_g=batched_g if args.teacher_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                gene_g=gene_g if args.teacher_type in ["UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                args=args
            )
            teacher_results = trainer.train(train_loader, val_loader, test_loader)
            all_results['teacher']['auprc'].append(teacher_results['test_metrics']['auprc'])
            all_results['teacher']['auroc'].append(teacher_results['test_metrics']['auroc'])
            all_results['teacher']['val_auprc'].append(teacher_results['val_avg_auprc'])

        elif args.mode == "student": 
            teacher_model = get_model(args.teacher_type, d_input, n_snps, n_genes, args, device)
            student_model = get_model(args.student_type, d_input, n_snps, n_genes, args, device)
            optimizer = get_optimizer(student_model, args.student_type, args.lr, args.use_adaptive_lr)
            criterion = KDLoss(temperature=args.temperature, alpha=args.alpha)
            trainer = KDTrainer(
                teacher_model=teacher_model,
                student_model=student_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=f'student_run_{run}',
                teacher_model_path=f'experiments/{args.teacher_model_exp_name if args.teacher_model_exp_name else args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_teacher_{args.teacher_type}_age{args.age_threshold}_T{args.teacher_model_temperature if args.teacher_model_temperature else args.temperature}_alpha{args.teacher_model_alpha if args.teacher_model_alpha else args.alpha}_es{args.teacher_model_es if args.teacher_model_es else args.n_early_stop}_eval{args.teacher_model_eval if args.teacher_model_eval else args.eval_interval}_lr{args.teacher_model_lr if args.teacher_model_lr else args.lr}_use_bgce{args.use_bgce}_use_lc{args.use_label_correction}_use_mixup{args.use_mixup}/teacher_run_{run}.pth',
                save_dir=exp_dir,
                snp_ids=torch.LongTensor(snp_ids) if args.teacher_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] or args.student_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                batched_g=batched_g if args.teacher_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] or args.student_type in ["UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                gene_g=gene_g if args.teacher_type in ["UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] or args.student_type in ["UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"] else None,
                args=args
            )
            student_results = trainer.train(train_loader, val_loader, test_loader)
            all_results['student']['auprc'].append(student_results['test_metrics']['auprc'])
            all_results['student']['auroc'].append(student_results['test_metrics']['auroc'])
            all_results['student']['val_auprc'].append(student_results['val_avg_auprc'])

    summary = {
        args.mode: {
            'test_auprc': f'{np.mean(all_results[args.mode]["auprc"])} ± {np.std(all_results[args.mode]["auprc"])}',
            'test_auroc': f'{np.mean(all_results[args.mode]["auroc"])} ± {np.std(all_results[args.mode]["auroc"])}',
            'val_auprc': f'{np.mean(all_results[args.mode]["val_auprc"])} ± {np.std(all_results[args.mode]["val_auprc"])}',
            'overfitting_ratio': f'{np.mean(all_results[args.mode]["val_auprc"])/np.mean(all_results[args.mode]["auprc"])}',
        },
        'remark' : {
            'remark': all_results['remark'] if all_results['remark'] else None
        }
    }
    with open(f'{exp_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nTraining {exp_dir.split('/')[-1]} completed. Summary saved.")
    return summary

if __name__ == '__main__':
    args = parse_train_args()
    check_cuda()
    set_random_seed()
    torch.use_deterministic_algorithms(True,warn_only=True) ##debug
    run_experiment(args)