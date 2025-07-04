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
import pickle
import json

from src.args import parse_train_args
from src.dataset import Dataset,MixDataset
from src.trainer import Trainer, Trainer_mix
from src.utils import set_random_seed, seed_worker, collate_fn_mix, split_by_age_mix, check_cuda, get_model, get_optimizer, check_data_mix
from src.teacher_models import MLP

def run_mix_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    d_input = np.load(f"{args.data_dir}/X/{info_df['sample_id'].iloc[0]}.npy").shape[0]
    
    splits_list = split_by_age_mix(
        info_df['label'].values, 
        info_df['age'].values, 
        args.age_threshold, 
        n_splits=args.n_runs
    )
    
    # 结果收集
    all_results = {
        'mix_model': {
            'auroc_ad': [], 'auprc_ad': [],
            'auroc_ms': [], 'auprc_ms': [],
            'auroc_avg': [], 'auprc_avg': [], 'val_auprc_avg': []
        },
        'remark': args.remark
    }
    
    for run in range(1, args.n_runs + 1):
        print(f"Run {run}/{args.n_runs}")
        
        labels = info_df['label'].copy()
        train_indices, val_indices, test_indices = splits_list[run-1]
        check_data_mix(info_df, train_indices, val_indices, test_indices, run, args.age_threshold)
        
        exp_dir = f'experiments/{args.exp_name}/{args.data_dir.split("/")[-1]}/{args.teacher_type}/{args.data_dir.split("/")[-1]}_{args.mode}_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.n_early_stop}_eval{args.eval_interval}_lr{args.lr}_use_bgce{args.use_bgce}_use_lc{args.use_label_correction}_use_mixup{args.use_mixup}'
        os.makedirs(exp_dir, exist_ok=True)
        
        train_dataset = MixDataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[train_indices].values, ages=info_df['age'].iloc[train_indices].values, labels=labels.iloc[train_indices].values, balanced_sampling=True)
        val_dataset = MixDataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[val_indices].values, ages=info_df['age'].iloc[val_indices].values, labels=labels.iloc[val_indices].values, balanced_sampling=False)
        test_dataset = MixDataset(feat_path=f"{args.data_dir}/X", sample_ids=info_df['sample_id'].iloc[test_indices].values, ages=info_df['age'].iloc[test_indices].values, labels=labels.iloc[test_indices].values, balanced_sampling=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=True, collate_fn=collate_fn_mix, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn_mix, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, collate_fn=collate_fn_mix, pin_memory=True)
        
        model = MLP(d_input=d_input, d_hidden=args.d_hidden, num_labels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()
        trainer = Trainer_mix(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=f'teacher_run_{run}',
            save_dir=exp_dir,
            args=args
        )
        
        results = trainer.train(train_loader, val_loader, test_loader)
        
        test_metrics = results['test_metrics']
        all_results['mix_model']['auroc_ad'].append(test_metrics['auroc_ad'])
        all_results['mix_model']['auprc_ad'].append(test_metrics['auprc_ad'])
        all_results['mix_model']['auroc_ms'].append(test_metrics['auroc_ms'])
        all_results['mix_model']['auprc_ms'].append(test_metrics['auprc_ms'])
        
        auroc_avg = (test_metrics['auroc_ad'] + test_metrics['auroc_ms']) / 2
        auprc_avg = (test_metrics['auprc_ad'] + test_metrics['auprc_ms']) / 2
        val_auprc_avg = results['val_avg_auprc']
        
        all_results['mix_model']['auroc_avg'].append(auroc_avg)
        all_results['mix_model']['auprc_avg'].append(auprc_avg)
        all_results['mix_model']['val_auprc_avg'].append(val_auprc_avg)
        
    summary = {
        'mix_model': {
            'test_auprc_ad': f'{np.mean(all_results["mix_model"]["auprc_ad"]):.4f} ± {np.std(all_results["mix_model"]["auprc_ad"]):.4f}',
            'test_auroc_ad': f'{np.mean(all_results["mix_model"]["auroc_ad"]):.4f} ± {np.std(all_results["mix_model"]["auroc_ad"]):.4f}',
            'test_auprc_ms': f'{np.mean(all_results["mix_model"]["auprc_ms"]):.4f} ± {np.std(all_results["mix_model"]["auprc_ms"]):.4f}',
            'test_auroc_ms': f'{np.mean(all_results["mix_model"]["auroc_ms"]):.4f} ± {np.std(all_results["mix_model"]["auroc_ms"]):.4f}',
            'test_auprc_avg': f'{np.mean(all_results["mix_model"]["auprc_avg"]):.4f} ± {np.std(all_results["mix_model"]["auprc_avg"]):.4f}',
            'test_auroc_avg': f'{np.mean(all_results["mix_model"]["auroc_avg"]):.4f} ± {np.std(all_results["mix_model"]["auroc_avg"]):.4f}',
            'val_auprc_avg': f'{np.mean(all_results["mix_model"]["val_auprc_avg"]):.4f} ± {np.std(all_results["mix_model"]["val_auprc_avg"]):.4f}',
            'overfitting_ratio': f'{np.mean(all_results["mix_model"]["val_auprc_avg"])/np.mean(all_results["mix_model"]["auprc_avg"]):.4f}',
        },
        'remark': {
            'remark': all_results['remark'] if all_results['remark'] else None
        }
    }
    
    with open(f'{exp_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    return summary

if __name__ == '__main__':
    args = parse_train_args()
    check_cuda()
    set_random_seed()
    torch.use_deterministic_algorithms(True)
    summary = run_mix_experiment(args)