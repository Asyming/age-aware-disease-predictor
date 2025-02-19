import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_experiment_info(dirname):
    try:
        parts = dirname.split('_')
        if len(parts) < 2 or 'es20' not in parts[-2]: # early stopping threshold
            return None
            
        disease = parts[0]  # ad/ms/uc
        mode = parts[1]     # teacher/student
        
        model = None
        teacher_model = None
        student_model = None
        age = None
        
        if mode == 'teacher':
            model = parts[2]
            for part in parts[3:]:
                if 'age' in part:
                    age = part.replace('age', '')
                    break
        else:  # student mode
            if len(parts) < 5:
                print(f"Student directory name too short: {dirname}")
                return None
            teacher_model = parts[2]
            student_model = parts[3]

            for part in parts[4:]:
                if 'age' in part:
                    age = part.replace('age', '')
                    break
        
        if age is None:
            print(f"No age found in directory name: {dirname}")
            return None
            
        return {
            'disease': disease,
            'mode': mode,
            'age': age,
            'model': model if mode == 'teacher' else f'{teacher_model}->{student_model}'
        }
    except Exception as e:
        print(f"Error parsing directory name '{dirname}': {str(e)}")
        return None

def extract_auprc(exp_dir):
    try:
        json_path = os.path.join(exp_dir, 'summary.json')
        if not os.path.exists(json_path):
            print(f"No summary.json found in {exp_dir}")
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        mode = 'teacher' if 'teacher_' in os.path.basename(exp_dir) else 'student'
        
        if mode not in data or data[mode]['auprc_mean'] is None:
            print(f"No valid AUPRC data for {mode} in {exp_dir}")
            return None
            
        return data[mode]['auprc_mean']
    except Exception as e:
        print(f"Error extracting AUPRC from {exp_dir}: {str(e)}")
        return None

def collect_results():
    results = {
        'ad_age65': {'baseline_teacher': [], 'baseline_student': [],
                     'age1_teacher': [], 'age1_student_age1': [], 'age1_student_mlp': [],
                     'age2_teacher': [], 'age2_student_age2': [], 'age2_student_mlp': []},
        'ad_age0': {'baseline_teacher': [], 'baseline_student': [],
                    'age1_teacher': [], 'age1_student_age1': [], 'age1_student_mlp': [],
                    'age2_teacher': [], 'age2_student_age2': [], 'age2_student_mlp': []},
        'ms_age65': {'baseline_teacher': [], 'baseline_student': [],
                    'age1_teacher': [], 'age1_student_age1': [], 'age1_student_mlp': [],
                    'age2_teacher': [], 'age2_student_age2': [], 'age2_student_mlp': []},
        'uc_age65': {'baseline_teacher': [], 'baseline_student': [],
                    'age1_teacher': [], 'age1_student_age1': [], 'age1_student_mlp': [],
                    'age2_teacher': [], 'age2_student_age2': [], 'age2_student_mlp': []}
    }
    
    exp_root = 'experiments'
    for dirname in os.listdir(exp_root):
        if not os.path.isdir(os.path.join(exp_root, dirname)):
            continue
            
        info = get_experiment_info(dirname)
        if info is None:
            continue
            
        auprc = extract_auprc(os.path.join(exp_root, dirname))
        if auprc is None:
            continue
        
        parts = dirname.split('_')
        
        result_key = f"{info['disease']}_age{info['age']}"
        if result_key not in results:
            continue
        
        if info['mode'] == 'teacher':
            if 'MLP' in parts[2] and 'AgeAware' not in parts[2]:
                exp_type = 'baseline_teacher'
            elif 'AgeAwareMLP1' in parts[2]:
                exp_type = 'age1_teacher'
            elif 'AgeAwareMLP2' in parts[2]:
                exp_type = 'age2_teacher'
            else:
                continue
        else:  # student mode
            teacher_model = parts[2]
            student_model = parts[3]
            
            if 'MLP' in teacher_model and 'AgeAware' not in teacher_model:
                exp_type = 'baseline_student'
            elif 'AgeAwareMLP1' in teacher_model:
                if 'MLP' in student_model and 'AgeAware' not in student_model:
                    exp_type = 'age1_student_mlp'
                else:
                    exp_type = 'age1_student_age1'
            elif 'AgeAwareMLP2' in teacher_model:
                if 'MLP' in student_model and 'AgeAware' not in student_model:
                    exp_type = 'age2_student_mlp'
                else:
                    exp_type = 'age2_student_age2'
            else:
                continue
        
        print(f"Processing {dirname} as {exp_type}")
        
        results[result_key][exp_type].append({
            'auprc': auprc,
            'mode': info['mode'],
            'model': info['model']
        })
    
    return results

def plot_results(results):
    groups = ['ad_age65', 'ad_age0', 'ms_age65', 'uc_age65']
    titles = ['AD (Age Threshold: 65)', 'AD (Age Threshold: 0)', 
              'MS (Age Threshold: 65)', 'UC (Age Threshold: 65)']
    exp_types = ['baseline_teacher', 'baseline_student',
                 'age1_teacher', 'age1_student_age1', 'age1_student_mlp',
                 'age2_teacher', 'age2_student_age2', 'age2_student_mlp']
    exp_labels = ['BT', 'BS', 'A1T', 'A1S1', 'A1S2', 'A2T', 'A2S1', 'A2S2']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    width = 0.7
    
    detailed_results = {}
    
    for ax, group, title in zip(axes, groups, titles):
        x = np.arange(len(exp_types))
        
        auprc_values = []
        detailed_results[group] = {}
        
        for exp_type in exp_types:
            if results[group][exp_type]:
                avg_auprc = np.mean([r['auprc'] for r in results[group][exp_type]])
                auprc_values.append(avg_auprc)
                detailed_results[group][exp_type] = {
                    'auprc': avg_auprc,
                    'details': [{'mode': r['mode'], 'model': r['model'], 'auprc': r['auprc']} 
                              for r in results[group][exp_type]]
                }
            else:
                auprc_values.append(0)
                detailed_results[group][exp_type] = {'auprc': 0, 'details': []}
        
        colors = ['blue', 'lightblue',
                 'red', 'lightcoral', 'pink',
                 'green', 'lightgreen', 'palegreen']
        
        bars = ax.bar(x, auprc_values, width, color=colors)
        
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, rotation=45)
        ax.set_ylabel('AUPRC')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    with open('experiment_details.json', 'w') as f:
        json.dump(detailed_results, f, indent=4)

def main():
    results = collect_results()
    plot_results(results)
    
    for disease in results:
        print(f"\n=== {disease.upper()} Dataset ===")
        for exp_type in results[disease]:
            print(f"\n{exp_type}:")
            for result in results[disease][exp_type]:
                print(f"Mode: {result['mode']}, Model: {result['model']}, AUPRC: {result['auprc']:.4f}")

if __name__ == '__main__':
    main()