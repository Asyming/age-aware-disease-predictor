import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/ageaware"))
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import inspect
from src.utils import split_by_age, set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Classical ML models")
    parser.add_argument("--exp_name", type=str, default='simple_classical')
    parser.add_argument("--ancestry", type=str, default='EUR')
    parser.add_argument("--age_threshold", type=int, default=65)
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc'])
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="LinearRegression", 
                      choices=["LinearRegression", "XGBoost", "LightGBM"])
    
    # Model-specific parameters
    parser.add_argument("--xgb_n_estimators", type=int, default=100)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    
    parser.add_argument("--lgb_n_estimators", type=int, default=100)
    parser.add_argument("--lgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--lgb_max_depth", type=int, default=5)
    
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--teacher_type", type=str, default="LinearRegression", 
                        choices=["LinearRegression", "XGBoost", "LightGBM"])
    parser.add_argument("--teacher_model_exp_name", type=str, default=None, 
                        help="Teacher model experiment name for loading")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Weight for hard labels vs soft labels (higher = more weight to hard labels)")
    parser.add_argument("--temperature", type=float, default=2.0, 
                        help="Temperature for softening probability distributions")
    
    return parser.parse_args()

def load_data(data_dir, info_df, train_indices, test_indices):
    """Load and preprocess data"""
    print("Loading data...")
    feature_dir = os.path.join(data_dir, 'X')
    feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])
    all_indices = np.concatenate([train_indices, test_indices])

    example_file_path = os.path.join(feature_dir, feature_files[all_indices[0]])
    example_feature = np.load(example_file_path)
    X = np.zeros((len(all_indices), example_feature.shape[0]))
    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(all_indices)}

    for i, idx in enumerate(all_indices):
        if i % 10000 == 0:
            print(f"Loading sample {i}/{len(all_indices)}...")
        
        file_path = os.path.join(feature_dir, feature_files[idx])
        feature = np.load(file_path)
        X[index_map[idx]] = feature
    
    y = info_df['label'].values[all_indices]
    
    train_indices_new = [index_map[idx] for idx in train_indices]
    test_indices_new = [index_map[idx] for idx in test_indices]
    
    X_train = X[train_indices_new]
    y_train = y[train_indices_new] 
    X_test = X[test_indices_new]
    y_test = y[test_indices_new]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Data loaded. Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler

def get_model(args):
    if args.model_type == "LinearRegression":
        return LinearRegression(
            n_jobs=-1
        )
    
    elif args.model_type == "XGBoost":
        return xgb.XGBRegressor(
            # n_estimators=args.xgb_n_estimators,
            # max_depth=args.xgb_max_depth,
            # learning_rate=args.xgb_learning_rate,
            device='cuda',
            n_jobs=-1
        )
    
    elif args.model_type == "LightGBM":
        return lgb.LGBMRegressor(
            # n_estimators=args.lgb_n_estimators,
            # max_depth=args.lgb_max_depth,
            # learning_rate=args.lgb_learning_rate,
            device='gpu',
            n_jobs=-1
        )

def evaluate_model(model, X, y, name='Test set'):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_prob = scaler.fit_transform(model.predict(X).reshape(-1, 1)).flatten()
    y_prob = sigmoid(y_prob)

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

def load_teacher_model(args, run, data_name):
    if args.teacher_model_exp_name is None:
        return None
    model_path = os.path.join(
        'experiments', 
        args.teacher_model_exp_name,
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

def get_soft_labels(teacher_model, X, temperature=1.0):
    preds = teacher_model.predict(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    soft_labels = scaler.fit_transform(preds.reshape(-1, 1)).flatten()/temperature
    soft_labels = sigmoid(soft_labels)
    print(f'soft_labels.shape: {soft_labels.shape}')
    print(f'soft_labels: {soft_labels}')
    print(f'min(soft_labels): {min(soft_labels)}')
    print(f'max(soft_labels): {max(soft_labels)}')
    return soft_labels

def train_with_soft_labels(model, X_train, y_train, soft_labels=None, alpha=0.5, model_type="LinearRegression"):
    if soft_labels is None or alpha == 1.0:
        model.fit(X_train, y_train)
        return model

    combined_targets = alpha * y_train + (1-alpha) * soft_labels.flatten()
    model.fit(X_train, combined_targets)
    return model

def run_experiment(args):
    data_name = args.data_dir.split("/")[-1]
    
    if args.mode == "teacher":
        exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.model_type}/{data_name}_{args.mode}_{args.model_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}'
    elif args.mode == "student":
        exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.teacher_type}/{data_name}_{args.mode}_{args.teacher_type}_{args.model_type}_age{args.age_threshold}_T{args.temperature}_alpha{args.alpha}'
    
    os.makedirs(exp_dir, exist_ok=True)
    
    info_df = pd.read_csv(f"{args.data_dir}/sample_info.csv")
    info_df = info_df[(info_df['ancestry'] == args.ancestry) & (info_df['age'] >= 40) & (info_df['age'] <= 70)].reset_index(drop=True)
    splits_list = split_by_age(info_df['label'].values, info_df['age'].values, args.age_threshold, n_splits=args.n_runs)
    
    all_results = {
        'test_auprc': []
    }
    
    for run in range(1, args.n_runs + 1):
        print(f"\n{'-'*40}")
        print(f"Running {run}/{args.n_runs}")
        print(f"{'-'*40}")
        
        train_indices, val_indices, test_indices = splits_list[run-1]
        train_indices = np.concatenate([train_indices, val_indices])
        X_train, y_train, X_test, y_test, scaler = load_data(
            args.data_dir, info_df, train_indices, test_indices
        )
        
        teacher_model = None
        soft_labels = None
        
        if args.mode == "student":
            teacher_model = load_teacher_model(args, run, data_name)
            if teacher_model is not None:
                soft_labels = get_soft_labels(
                    teacher_model, 
                    X_train, 
                    temperature=args.temperature
                )
        
        print(f"\nTraining {args.model_type} model...")
        
        model = get_model(args)
        if args.mode == "teacher" or soft_labels is None:
            model.fit(X_train, y_train)
        else:
            model = train_with_soft_labels(
                model, 
                X_train, 
                y_train, 
                soft_labels=soft_labels, 
                alpha=args.alpha,
                model_type=args.model_type
            )

        test_metrics = evaluate_model(model, X_test, y_test, name='Test set')
        
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
    warnings.filterwarnings("ignore")
    args = parse_args()
    set_random_seed()
    run_experiment(args)