import os
import sys
sys.path.append(os.path.abspath("/data3/lihan/projects/zisen/ageaware"))
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
    parser.add_argument("--data_dir", type=str, default='./data/ad', choices=['./data/ad','./data/ms','./data/uc', './data/ad_new'])
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="LinearRegression", choices=["LinearRegression", "XGBoostRegressor", "LightGBMRegressor", "XGBoostClassifier", "LightGBMClassifier"])
    
    # Model-specific parameters
    parser.add_argument("--xgb_learning_rate", type=float, default=0.0001)
    parser.add_argument("--xgb_n_estimators", type=int, default=5000)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_max_leaves", type=int, default=0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.01) # L1
    parser.add_argument("--xgb_reg_lambda", type=float, default=0.1) # L2
    parser.add_argument("--xgb_scale_pos_weight", type=float, default=30)
    parser.add_argument("--lgb_n_estimators", type=int, default=500)
    parser.add_argument("--lgb_learning_rate", type=float, default=0.001)
    parser.add_argument("--lgb_max_depth", type=int, default=6)
    parser.add_argument("--es_rounds", type=int, default=1000)
    
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
            n_jobs=8
        )
    
    elif args.model_type == "XGBoostRegressor":
        return xgb.XGBRegressor(
            objective='binary:logistic',
            learning_rate=args.xgb_learning_rate,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            early_stopping_rounds=args.es_rounds,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            device='gpu',
            n_jobs=8
        )
    
    elif args.model_type == "LightGBMRegressor":
        return lgb.LGBMRegressor(
            objective='binary',
            #objective='cross_entropy',
            learning_rate=args.lgb_learning_rate,
            n_estimators=args.lgb_n_estimators,
            max_depth=args.lgb_max_depth,
            device='gpu',
            n_jobs=8
        )
    
    elif args.model_type == "XGBoostClassifier":
        return xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric="aucpr",
            scale_pos_weight=args.xgb_scale_pos_weight,
            learning_rate=args.xgb_learning_rate,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            max_leaves=args.xgb_max_leaves,
            early_stopping_rounds=args.es_rounds,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            device='gpu',
            n_jobs=8
        )   
    
    elif args.model_type == "LightGBMClassifier":
        return lgb.LGBMClassifier(
            objective='binary',
            n_estimators=args.lgb_n_estimators,
            max_depth=args.lgb_max_depth,
            learning_rate=args.lgb_learning_rate,
            device='gpu',
            n_jobs=8
        )
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

def train_model(model, X_train, y_train, X_val=None, y_val=None, es_rounds=50, model_type="LinearRegression"):
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
    elif model_type == "XGBoostClassifier" and X_val is not None and y_val is not None:
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 verbose=True)
    elif model_type == "LightGBMClassifier" and X_val is not None and y_val is not None:
        callbacks = [lgb_es(stopping_rounds=es_rounds, verbose=True)]
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 eval_metric="average_precision",
                 callbacks=callbacks)
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X, y, name='Test set'):
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        scaler = MinMaxScaler()
        y_prob = scaler.fit_transform(model.predict(X).reshape(-1, 1)).flatten()
    
    # debug
    print(f'y_prob.shape: {y_prob.shape}')
    print(f'y_prob sample: {y_prob[:5]}')
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
    exp_dir = f'experiments/{args.exp_name}/{data_name}/{args.model_type}/{data_name}_{args.model_type}_age{args.age_threshold}'
    
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
        
        print(f"Data prepared. Running {run}. Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        print(f"\nTraining {args.model_type} model...")
        
        model = get_model(args)
        model = train_model(
            model, 
            X_train, 
            y_train,
            X_val=X_val,
            y_val=y_val,
            es_rounds=args.es_rounds,
            model_type=args.model_type
        )

        test_metrics = evaluate_model(model, X_test, y_test, name='Test set')
        
        # Save model
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