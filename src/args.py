import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Train Age-Aware Teacher or Student Models")
    ## experiment settings
    parser.add_argument("--exp_name", type=str, default='test1')
    parser.add_argument("--teacher_model_exp_name", type=str, default=None)
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=65)
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc','./data/af'])
    parser.add_argument("--age_range", type=str, default='40-70', help='age range: min-max')
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument("--teacher_type", type=str, default="MLP", choices=["MLP", "ctrMLP", "AgeAwareMLP1", "AgeAwareMLP2", "UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"])
    parser.add_argument("--student_type", type=str, default=None, choices=["MLP", "ctrMLP", "AgeAwareMLP1", "AgeAwareMLP2", "UGP_v1", "UGP_v2", "UGP_v3", "ctrUGP_v1", "AgeUGP_v1", "AgeUGP_v2"])
    parser.add_argument("--remark", type=str, default=None)
    ## model settings
    parser.add_argument("--num_workers", type=int, default=8)
    # Teacher GNN models: lr = 1e-5 performs better
    # Teacher other models: lr = 1e-4 performs better
    # Student models: lr = teacher_model_lr * 10 performs better
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--n_gnn_layers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_early_stop", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2)

    parser.add_argument("--teacher_model_lr", type=float, default=1e-4)
    parser.add_argument("--teacher_model_es", type=int, default=20)
    parser.add_argument("--teacher_model_eval", type=int, default=50)
    parser.add_argument("--teacher_model_alpha", type=float, default=0.1)
    parser.add_argument("--teacher_model_temperature", type=float, default=2)
    parser.add_argument("--snp_dropout", type=float, default=0.5)
    parser.add_argument("--gene_dropout", type=float, default=0.0)
    # ctr hyper-parameters
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--pos_tau", type=float, default=0.99)
    parser.add_argument("--neg_tau", type=float, default=0.70)
    parser.add_argument("--neg_rate", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=1.0)
    ## ablation study
    parser.add_argument("--age1_use_consist", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age1_use_adversarial", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_consist", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_ageloss", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--age2_use_disentangle", type=lambda x: x.lower() == 'true', default=True)
    ## auxiliary tools
    parser.add_argument("--use_adaptive_lr", action='store_true') 
    parser.add_argument("--use_cumulative_rate", action='store_true')
    parser.add_argument("--use_label_correction", action='store_true')
    parser.add_argument("--pseudo_label_interval", type=int, default=20)
    parser.add_argument("--pseudo_label_start_step", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--use_bgce", action='store_true')
    parser.add_argument("--bgce_q", type=float, default=0.6) # 0.6
    parser.add_argument("--use_mixup", action='store_true')
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--use_elr", action='store_true')
    return parser.parse_args()