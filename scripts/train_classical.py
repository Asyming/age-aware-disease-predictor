import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/ageaware"))
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping as lgb_es
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
from src.utils import split_by_age, set_random_seed
import torch
from src.teacher_models import MLP, AgeAwareMLP1, AgeAwareMLP2

def parse_args():
    parser = argparse.ArgumentParser(description="Classical ML models")
    parser.add_argument("--exp_name", type=str, default='simple_classical')
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=65)
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc'])
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="LinearRegression", choices=["LinearRegression", "XGBoostRegressor", "LightGBMRegressor", "XGBoostClassifier", "LightGBMClassifier"])
    
    # Model-specific parameters
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--xgb_n_estimators", type=int, default=100)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    
    parser.add_argument("--lgb_n_estimators", type=int, default=100)
    parser.add_argument("--lgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--lgb_max_depth", type=int, default=5)
    parser.add_argument("--es_rounds", type=int, default=20)
    
    parser.add_argument("--mode", type=str, required=True, choices=["teacher", "student"])
    parser.add_argument("--teacher_type", type=str, default="LinearRegression", choices=["LinearRegression", "XGBoostRegressor", "LightGBMRegressor"])
    parser.add_argument("--teacher_model_exp_name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1, help="higher = more weight to hard labels")
    parser.add_argument("--temperature", type=float, default=2)
    
    # Train4 teacher model parameters
    parser.add_argument("--train4_teacher_exp_name", type=str, default=None)
    parser.add_argument("--train4_teacher_type", type=str, default="MLP", choices=["MLP", "AgeAwareMLP1", "AgeAwareMLP2"])
    parser.add_argument("--train4_teacher_es", type=int, default=20)
    parser.add_argument("--train4_teacher_eval", type=int, default=50)
    parser.add_argument("--train4_teacher_lr", type=float, default=1e-4)
    
    return parser.parse_args()

def load_all_data(data_dir, info_df):
    print("Loading all data...")
    feature_dir = os.path.join(data_dir, 'X')
    feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])
    
    example_file_path = os.path.join(feature_dir, feature_files[0])
    example_feature = np.load(example_file_path)
    X = np.zeros((len(feature_files), example_feature.shape[0]))
    
    for i in range(len(feature_files)):
        if i % 10000 == 0:
            print(f"Loading sample {i}/{len(feature_files)}...")
        
        file_path = os.path.join(feature_dir, feature_files[i])
        feature = np.load(file_path, mmap_mode='r')
        X[i] = feature
    
    y = info_df['label'].values
    
    print(f"All data loaded. X shape: {X.shape}")
    return X, y

def get_model(args):
    if args.model_type == "LinearRegression":
        return LinearRegression(
            n_jobs=-1
        )
    
    elif args.model_type == "XGBoostRegressor":
        return xgb.XGBRegressor(
            learning_rate=args.lr,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            # device='cuda' # 会爆内存
            early_stopping_rounds=args.es_rounds,
            n_jobs=-1
        )
    
    elif args.model_type == "LightGBMRegressor":
        return lgb.LGBMRegressor(
            learning_rate=args.lr,
            objective='cross_entropy', # 不太合适
            n_estimators=args.lgb_n_estimators,
            max_depth=args.lgb_max_depth,
            learning_rate=args.lgb_learning_rate,
            device='gpu',
            n_jobs=-1
        )
    
    elif args.model_type == "XGBoostClassifier":
        return xgb.XGBClassifier(
            objective='binary:',
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            n_jobs=-1
        )   
    
    elif args.model_type == "LightGBMClassifier":
        return lgb.LGBMClassifier(
            objective='binary',
            n_estimators=args.lgb_n_estimators,
            max_depth=args.lgb_max_depth,
            learning_rate=args.lgb_learning_rate,
            device='gpu',
            n_jobs=-1
        )
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

# classical teacher model
def load_teacher_model(args, run, data_name):
    model_path = os.path.join(
        'experiments', 
        args.teacher_model_exp_name if args.teacher_model_exp_name is not None else args.exp_name,
        data_name,
        args.teacher_type,
        f'{data_name}_teacher_{args.teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}',
        f'run_{run}',
        'model.pkl'
    )
    print(f"\nLoading teacher model: {model_path}")
    model_data = joblib.load(model_path)
    teacher_model = model_data['model']
    return teacher_model

def get_soft_labels(teacher_model, X, temperature=2.0):
    # preds close to 0. sigmoid directly seems to perform worse. scale to -1~1 first.
    preds = teacher_model.predict(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    soft_labels = scaler.fit_transform(preds.reshape(-1, 1)).flatten()/temperature
    soft_labels = sigmoid(soft_labels)
    # debug
    print(f'soft_labels.shape: {soft_labels.shape}')
    print(f'Sample soft_labels: {soft_labels}')
    print(f'min(soft_labels): {min(soft_labels)}')
    print(f'max(soft_labels): {max(soft_labels)}')
    return soft_labels

# train4 teacher model
def load_train4_teacher(args, run, data_name):
    """Load teacher model from train4.py (.pth file)"""
    if args.train4_teacher_exp_name is None:
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(
        'experiments',
        args.train4_teacher_exp_name,
        data_name,
        args.train4_teacher_type,
        f'{data_name}_teacher_{args.train4_teacher_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}_es{args.train4_teacher_es}_eval{args.train4_teacher_eval}_lr{args.train4_teacher_lr}_run{run}',
        f'teacher_run_{run}.pth'
    )
    
    print(f"\nLoading train4 teacher model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return None

    example_file = os.path.join(args.data_dir, 'X', os.listdir(os.path.join(args.data_dir, 'X'))[0])
    d_input = np.load(example_file).shape[0]
    
    if args.train4_teacher_type == "MLP":
        model = MLP(d_input, d_hidden=64)
    elif args.train4_teacher_type == "AgeAwareMLP1":
        model = AgeAwareMLP1(d_input, d_hidden=64)
    elif args.train4_teacher_type == "AgeAwareMLP2":
        model = AgeAwareMLP2(d_input, d_hidden=64)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def get_train4_soft_labels(model, device, X_train, batch_size=128, temperature=2.0):
    """Get soft labels from train4 teacher model"""
    soft_labels = []
    
    with torch.no_grad():
        for i in range(0, len(X_train), batch_size):
            batch_end = min(i + batch_size, len(X_train))
            batch_X = torch.FloatTensor(X_train[i:batch_end]).to(device)
            
            outputs = model(batch_X)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            probs = torch.sigmoid(logits / temperature).cpu().numpy()
            soft_labels.append(probs)
    
    soft_labels = np.concatenate(soft_labels).flatten()
    # debug
    print(f'Train4 soft_labels.shape: {soft_labels.shape}')
    print(f'Sample soft_labels: {soft_labels[:10]}')
    print(f'min(soft_labels): {min(soft_labels)}')
    print(f'max(soft_labels): {max(soft_labels)}')
    
    return soft_labels

def train_with_soft_labels(model, X_train, y_train, X_val=None, y_val=None, soft_labels=None, alpha=0.1, es_rounds=20, verbose=100, model_type="LinearRegression"):
    if soft_labels is None:
        if model_type == "XGBoostRegressor" and X_val is not None and y_val is not None:
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=True)
        elif model_type == "LightGBMRegressor" and X_val is not None and y_val is not None:
            callbacks = [lgb_es(stopping_rounds=es_rounds, verbose=True)]
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     eval_metric="average_precision",
                     callbacks=callbacks)
        else:
            model.fit(X_train, y_train)
        return model

    combined_targets = alpha * y_train + (1-alpha) * soft_labels.flatten()
    if model_type == "XGBoostRegressor" and X_val is not None and y_val is not None:
        model.fit(X_train, combined_targets,
                 eval_set=[(X_val, y_val)],
                 verbose=True)
    elif model_type == "LightGBMRegressor" and X_val is not None and y_val is not None:
        callbacks = [lgb_es(stopping_rounds=es_rounds, verbose=True)]
        model.fit(X_train, combined_targets,
                 eval_set=[(X_val, y_val)],
                 eval_metric="average_precision",
                 callbacks=callbacks)
    else:
        model.fit(X_train, combined_targets)
    
    return model

def evaluate_model(model, X, y, name='Test set'):
    # compatible with soft_labels from classical or train4
    scaler = MinMaxScaler()
    y_prob = scaler.fit_transform(model.predict(X).reshape(-1, 1)).flatten()
    # debug
    print(f'y_prob.shape: {y_prob.shape}')
    print(f'y_prob: {y_prob}')
    print(f'min(y_prob): {min(y_prob)}')
    print(f'max(y_prob): {max(y_prob)}')
    y_pred = (y_prob > 0.5).astype(int)
    auprc = average_precision_score(y, y_prob)

    print(classification_report(y, y_pred))
    print(f"AUPRC: {auprc:.4f}")
    
    return {
        'auprc': auprc,
        'predictions': y_prob,
        'pred_labels': y_pred,
        'true_labels': y
    }

def run_experiment(args):
    data_name = args.data_dir.split("/")[-1]
    if args.mode == "teacher":
        exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.model_type}/{data_name}_{args.mode}_{args.model_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}'
    elif args.mode == "student":
        if args.train4_teacher_exp_name is not None:
            exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.train4_teacher_type}/{data_name}_{args.mode}_{args.train4_teacher_type}_{args.model_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}'
        else:
            exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.teacher_type}/{data_name}_{args.mode}_{args.teacher_type}_{args.model_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}'
    
    os.makedirs(exp_dir, exist_ok=True)
    
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[(info_df['ancestry'] == args.ancestry) & (info_df['age'] >= 40) & (info_df['age'] <= 70)].reset_index(drop=True)
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold, n_splits=args.n_runs)
    
    all_X, all_y = load_all_data(args.data_dir, info_df)
    all_results = {
        'test_auprc': []
    }
    
    for run in range(1, args.n_runs + 1):
        print(f"\n{'-'*40}")
        print(f"Running {run}/{args.n_runs}")
        print(f"{'-'*40}")
        
        train_indices, val_indices, test_indices = splits_list[run-1]
        
        X_train = all_X[train_indices]
        y_train = all_y[train_indices]
        X_val = all_X[val_indices]
        y_val = all_y[val_indices]
        X_test = all_X[test_indices]
        y_test = all_y[test_indices]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        print(f"Data prepared for run {run}. Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        teacher_model = None
        soft_labels = None
        train4_model_info = None

        # train4 soft_labels
        if args.mode == "student" and args.train4_teacher_exp_name is not None:
            train4_model_info = load_train4_teacher(args, run, data_name)
            if train4_model_info is not None:
                train4_model, device = train4_model_info
                soft_labels = get_train4_soft_labels(
                    train4_model, 
                    device,
                    X_train, 
                    temperature=args.temperature
                )
        # classical soft_labels
        elif args.mode == "student" and args.teacher_model_exp_name is not None:
            teacher_model = load_teacher_model(args, run, data_name)
            if teacher_model is not None:
                soft_labels = get_soft_labels(
                    teacher_model, 
                    X_train, 
                    temperature=args.temperature
                )
        
        print(f"\nTraining {args.model_type} model...")
        
        model = get_model(args)
        model = train_with_soft_labels(
            model, 
            X_train, 
            y_train,
            X_val=X_val,
            y_val=y_val,
            soft_labels=soft_labels if args.mode == "student" else None,
            alpha=args.alpha,
            es_rounds=args.es_rounds,
            model_type=args.model_type
        )

        test_metrics = evaluate_model(model, X_test, y_test, name='Test set')
        
        # save model
        run_dir = os.path.join(f'{exp_dir}/run_{run}')
        os.makedirs(run_dir, exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'test_metrics': test_metrics
        }, os.path.join(run_dir, f'model.pkl'))
        
        all_results['test_auprc'].append(test_metrics['auprc'])
    
    summary = {
        'test_auprc_mean': np.mean(all_results['test_auprc']),
        'test_auprc_std': np.std(all_results['test_auprc'])
    }
    
    print(f"\n{'-'*40}")
    print(f"Experiment summary - {args.model_type}")
    print(f"{'-'*40}")
    print(f"Test set AUPRC: {summary['test_auprc_mean']:.4f} ± {summary['test_auprc_std']:.4f}")
    
    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nExperiment completed. Results saved in {exp_dir}")
    return summary

if __name__ == '__main__':
    args = parse_args()
    set_random_seed()
    run_experiment(args)