import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os
import re
import argparse

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Generate experiment result tables for specific diseases')
parser.add_argument('--disease', type=str, choices=['ad', 'ms', 'uc', 'all'], default='all', 
                    help='Disease to analyze (ad, ms, uc, or all)')
args = parser.parse_args()

# Define function to collect all experiment results
def collect_experiment_results(base_dir='experiments'):
    results = []
    
    # Recursively find all summary.json files
    for summary_file in glob.glob(f'{base_dir}/**/summary.json', recursive=True):
        exp_path = os.path.dirname(summary_file)
        
        # Parse experiment information from path
        path_parts = exp_path.split('/')
        exp_name = path_parts[1]  # Experiment group name
        disease = path_parts[2]   # Disease type
        model_type = path_parts[3]  # Model type (teacher's model type)
        
        # Extract configuration information (from filename)
        config_info = path_parts[-1]
        
        # Distinguish between teacher and different types of student
        if 'teacher_' in config_info or '_teacher_' in config_info:
            mode = 'teacher'
            student_type = None
        else:
            # For student paths like: ad_student_AgeAwareMLP2_AgeAwareMLP2_age65...
            match = re.search(r'student_([^_]+)_([^_]+)_age', config_info)
            if match:
                teacher_model = match.group(1)
                student_model = match.group(2)
                
                # Handle the special case for student_enhanced_lr3
                if exp_name == 'student_enhanced_lr3':
                    if student_model == 'MLP':
                        mode = 'student2'
                    elif 'AgeAwareMLP1' in student_model or 'AgeAwareMLP2' in student_model:
                        if student_model == teacher_model:
                            mode = 'student1'
                        else:
                            # For this experiment, if models don't match but both are age-aware
                            # we consider it student2
                            mode = 'student2'
                    else:
                        mode = 'student2'  # Default
                else:
                    # If student model matches teacher model, it's student1
                    if student_model == teacher_model:
                        mode = 'student1'
                    # If student model is MLP while teacher is AgeAwareMLP, it's student2
                    elif student_model == 'MLP' or (('AgeAwareMLP' in teacher_model) and not ('AgeAwareMLP' in student_model)):
                        mode = 'student2'
                    else:
                        mode = 'student1'  # Default to student1 if can't determine
                
                student_type = student_model
            else:
                # If pattern doesn't match, try to make a reasonable guess
                if 'student_' in config_info:
                    if 'MLP_' in config_info and 'AgeAwareMLP' in model_type:
                        mode = 'student2'
                    elif model_type in config_info:
                        mode = 'student1'
                    else:
                        mode = 'student2'  # Default
                else:
                    mode = 'student2'  # Default
                student_type = 'unknown'
        
        # Read results
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Determine which key to extract based on mode
        result_key = 'teacher' if mode == 'teacher' else 'student'
        
        # Record experiment information
        if result_key in summary:
            results.append({
                'experiment_group': exp_name,
                'disease': disease,
                'model_type': model_type,
                'student_type': student_type,
                'mode': mode,
                'auprc_mean': summary[result_key].get('auprc_mean', None),
                'auprc_std': summary[result_key].get('auprc_std', None),
                'path': exp_path,
                'config_info': config_info  # Save for debugging
            })
    
    return pd.DataFrame(results)

# Collect results
results_df = collect_experiment_results()

# Filter results by disease if specified
if args.disease != 'all':
    results_df = results_df[results_df['disease'] == args.disease]
    print(f"Analyzing results for {args.disease} disease only")
else:
    print("Analyzing results for all diseases")

# Debug: Print detected modes and paths
print(f"\nDetected modes and corresponding paths for {args.disease if args.disease != 'all' else 'all diseases'}:")
for mode in ['teacher', 'student1', 'student2']:
    sample_rows = results_df[results_df['mode'] == mode].head(3)
    if not sample_rows.empty:
        print(f"\n{mode} examples:")
        for _, row in sample_rows.iterrows():
            print(f"  - {row['config_info']} -> {row['mode']}")

# Process each disease separately or all together
diseases_to_process = [args.disease] if args.disease != 'all' else results_df['disease'].unique()

for disease_to_process in diseases_to_process:
    # Skip if not processing a specific disease
    if disease_to_process == 'all':
        # Create comprehensive tables for all diseases combined
        disease_results = results_df
        disease_prefix = ""
    else:
        # Filter for the current disease
        disease_results = results_df[results_df['disease'] == disease_to_process]
        disease_prefix = f"{disease_to_process}_"
        
        if disease_results.empty:
            print(f"No results found for disease: {disease_to_process}")
            continue
    
    print(f"\n\nProcessing results for {disease_prefix}diseases")
    
    # Create comprehensive table with experiment results (mean and std)
    print(f"\nComprehensive Table with Experiment Results for {disease_prefix}diseases:")
    
    # Create DataFrame to store the comprehensive results
    rows_list = []
    index_tuples = []
    
    # For each experiment group
    for group in sorted(disease_results['experiment_group'].unique()):
        group_df = disease_results[disease_results['experiment_group'] == group]
        
        # If processing all diseases, include disease in grouping
        if disease_to_process == 'all':
            # For each disease
            for disease in sorted(group_df['disease'].unique()):
                disease_df = group_df[group_df['disease'] == disease]
                
                # For each model type
                for model in sorted(disease_df['model_type'].unique()):
                    model_df = disease_df[disease_df['model_type'] == model]
                    
                    # Create row for this experiment
                    row_data = {}
                    
                    # Add teacher, student1, student2 results in that order
                    for mode in ['teacher', 'student1', 'student2']:
                        mode_df = model_df[model_df['mode'] == mode]
                        if not mode_df.empty:
                            # Get first matching result
                            row = mode_df.iloc[0]
                            row_data[f'{mode}_mean'] = row['auprc_mean']
                            row_data[f'{mode}_std'] = row['auprc_std']
                        else:
                            # If no result for this mode, use NaN
                            row_data[f'{mode}_mean'] = np.nan
                            row_data[f'{mode}_std'] = np.nan
                    
                    # Add row data to list
                    rows_list.append(row_data)
                    # Add index tuple for this row
                    index_tuples.append((group, disease, model))
        else:
            # For each model type
            for model in sorted(group_df['model_type'].unique()):
                model_df = group_df[group_df['model_type'] == model]
                
                # Create row for this experiment
                row_data = {}
                
                # Add teacher, student1, student2 results in that order
                for mode in ['teacher', 'student1', 'student2']:
                    mode_df = model_df[model_df['mode'] == mode]
                    if not mode_df.empty:
                        # Get first matching result
                        row = mode_df.iloc[0]
                        row_data[f'{mode}_mean'] = row['auprc_mean']
                        row_data[f'{mode}_std'] = row['auprc_std']
                    else:
                        # If no result for this mode, use NaN
                        row_data[f'{mode}_mean'] = np.nan
                        row_data[f'{mode}_std'] = np.nan
                
                # Add row data to list
                rows_list.append(row_data)
                # Add index tuple for this row
                if disease_to_process == 'all':
                    index_tuples.append((group, model))
                else:
                    index_tuples.append((group, model))
    
    # Create a MultiIndex
    if disease_to_process == 'all':
        multi_idx = pd.MultiIndex.from_tuples(index_tuples, names=['Experiment Group', 'Disease', 'Model Type'])
    else:
        multi_idx = pd.MultiIndex.from_tuples(index_tuples, names=['Experiment Group', 'Model Type'])
    
    # Create the DataFrame with the MultiIndex
    all_results = pd.DataFrame(rows_list, index=multi_idx)
    
    # Reorder columns to ensure teacher, student1, student2 order
    ordered_columns = []
    for mode in ['teacher', 'student1', 'student2']:
        if f'{mode}_mean' in all_results.columns:
            ordered_columns.append(f'{mode}_mean')
        if f'{mode}_std' in all_results.columns:
            ordered_columns.append(f'{mode}_std')
    
    all_results = all_results[ordered_columns]
    
    # Format the table for display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(all_results)
    
    # Create a formatted version with mean±std
    print(f"\nFormatted Table (mean±std) for {disease_prefix}diseases:")
    formatted_results = pd.DataFrame(index=all_results.index)
    
    for mode in ['teacher', 'student1', 'student2']:
        mean_col = f'{mode}_mean'
        std_col = f'{mode}_std'
        
        if mean_col in all_results.columns and std_col in all_results.columns:
            formatted_results[mode] = all_results.apply(
                lambda row: f"{row[mean_col]:.4f}±{row[std_col]:.4f}" if not pd.isna(row[mean_col]) else "",
                axis=1
            )
    
    print(formatted_results)
    
    # Save to CSV with disease prefix
    csv_filename = f"{disease_prefix}all_experiment_results.csv"
    formatted_csv_filename = f"{disease_prefix}formatted_experiment_results.csv"
    
    all_results.to_csv(csv_filename)
    formatted_results.to_csv(formatted_csv_filename)
    print(f"\nDetailed results saved to '{csv_filename}' and '{formatted_csv_filename}'")

# Create ablation experiment comparison chart
if any(results_df['experiment_group'].str.startswith('ab_')):
    plt.figure(figsize=(15, 10))
    
    # Filter ablation experiment results
    ablation_results = results_df[results_df['experiment_group'].str.startswith('ab_')].copy()
    
    # Arrange experiment groups in order - using .loc to avoid the warning
    ablation_order = ['ab_age1_adv', 'ab_age1_consist', 'ab_age2_ageloss', 
                      'ab_age2_disentangle', 'ab_age2_consist', 'ab_age2_ageloss_disentangle']
    ablation_results.loc[:, 'group_order'] = pd.Categorical(
        ablation_results['experiment_group'],
        categories=ablation_order,
        ordered=True
    )
    ablation_results = ablation_results.sort_values('group_order')

# Summary report
print("\n\nExperiment Results Summary Report")
print("=" * 50)
print(f"Total experiment groups: {len(results_df['experiment_group'].unique())}")
print(f"Disease types: {', '.join(results_df['disease'].unique())}")
print(f"Model types: {', '.join(results_df['model_type'].unique())}")

# Find the best model for each disease
for disease in sorted(results_df['disease'].unique()):
    print(f"\nBest performing models for {disease} disease:")
    for mode in ['teacher', 'student1', 'student2']:
        subset = results_df[(results_df['disease'] == disease) & (results_df['mode'] == mode)]
        if not subset.empty:
            best_row = subset.loc[subset['auprc_mean'].idxmax()]
            print(f"- {mode}: {best_row['model_type']} (Experiment Group: {best_row['experiment_group']}) - AUPRC: {best_row['auprc_mean']:.4f}±{best_row['auprc_std']:.4f}")
