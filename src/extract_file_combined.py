import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse
from pathlib import Path
from typing import Dict

DISEASE_PALETTE = {
    'ad': {
        'teacher_mlp1': '#08306B', 
        'student_mlp1': '#4292C6', 
        'teacher_mlp2': '#00441B',
        'student_mlp2': '#41AB5D'   
    },
    'ms': {
        'teacher_mlp1': '#8C2D04',
        'student_mlp1': '#FE9929',
        'teacher_mlp2': '#7A0177',
        'student_mlp2': '#AE017E' 
    },
    'uc': {
        'teacher_mlp1': '#3F007D',
        'student_mlp1': '#807DBA',
        'teacher_mlp2': '#662506',
        'student_mlp2': '#CC4C02',
    }
}

def parse_experiment(exp_name: str) -> dict:
    parts = exp_name.split('_')
    
    arch_version = 1 if 'AgeAwareMLP1' in exp_name else 2
    
    return {
        'disease': parts[0],
        'model_type': parts[1],
        'arch_version': arch_version,
        'teacher_arch': parts[2],
        'student_arch': parts[3] if parts[1] == 'student' else None,
        'age_threshold': int(parts[-3].replace('age', '')),
        'es': int(parts[-2].replace('es', '')),
        'run': parts[-1]
    }

def load_and_filter_data(json_path: str, config: dict) -> Dict:
    with open(json_path) as f:
        raw_data = json.load(f)
    
    filtered = {}
    for exp_name, exp_data in raw_data.items():
        params = parse_experiment(exp_name)
        
        params['full_type'] = f"{params['model_type']}_mlp{params['arch_version']}"

        if not (params['disease'] in config['diseases'] and
                params['model_type'] in config['model_types'] and
                params['age_threshold'] in config['age_thresholds'] and
                params['es'] in config['es_values']):
            continue

        valid_ages = {}
        for age_str, probs_data in exp_data.items():
            try:
                if isinstance(probs_data, dict):
                    p10_mean = probs_data.get('P10', {}).get('mean')
                    p10_std = probs_data.get('P10', {}).get('std')
                    p11_mean = probs_data.get('P11', {}).get('mean')
                    p11_std = probs_data.get('P11', {}).get('std')
                
                elif isinstance(probs_data, list) and len(probs_data) >= 2:
                    p10_mean = float(probs_data[1][0]) 
                    p10_std = 0.0
                    p11_mean = float(probs_data[1][1])
                    p11_std = 0.0
                
                else:
                    continue

                if all(not np.isnan(x) for x in [p10_mean, p11_mean]):
                    valid_ages[float(age_str.strip('"'))] = {
                        'p10': (p10_mean, p10_std),
                        'p11': (p11_mean, p11_std)
                    }
            
            except (KeyError, IndexError, ValueError, TypeError) as e:
                print(f"Failure: {exp_name}@{age_str} - {str(e)}")
                continue

        if valid_ages:
            filtered[exp_name] = {
                'params': params,
                'data': valid_ages
            }
    
    return filtered

def create_hierarchical_legend(config):
    import matplotlib.patches as mpatches
    
    legend_elements = []
    
    arch_legend = [
        # mpatches.Patch(color='gray', label='MLP1 (Square Marker)'),
        # mpatches.Patch(color='gray', label='MLP2 (Circle Marker)')
    ]
    
    role_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='Teacher'),
        Line2D([0], [0], color='black', linestyle='--', label='Student')
    ]
    
    disease_boxes = []
    for d in config['diseases']:
        box = plt.Line2D([0], [0], 
                        color=DISEASE_PALETTE[d]['teacher_mlp1'],
                        marker='s', linestyle='-',
                        label=f"{d.upper()} MLP1 Teach")
        disease_boxes.append(box)
        
        box = plt.Line2D([0], [0], 
                        color=DISEASE_PALETTE[d]['student_mlp1'],
                        marker='s', linestyle='--',
                        label=f"{d.upper()} MLP1 Stu")
        disease_boxes.append(box)
        
        box = plt.Line2D([0], [0], 
                        color=DISEASE_PALETTE[d]['teacher_mlp2'],
                        marker='o', linestyle='-',
                        label=f"{d.upper()} MLP2 Teach") 
        disease_boxes.append(box)
        
        box = plt.Line2D([0], [0], 
                        color=DISEASE_PALETTE[d]['student_mlp2'],
                        marker='o', linestyle='--',
                        label=f"{d.upper()} MLP2 Stu")
        disease_boxes.append(box)
    
    return [*arch_legend, *role_legend, *disease_boxes]

def plot_with_error_bars(filtered_data: Dict, config: dict):
    """带误差线的增强可视化"""
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    style_config = {
        'teacher_mlp1': {'ls': '-', 'marker': 'o', 'markersize': 2, 'capsize': 2.5, 'elinewidth': 0.8, 'alpha':0.6},
        'student_mlp1': {'ls': '--', 'marker': 'o', 'markersize': 2, 'capsize': 2.5, 'elinewidth': 0.8, 'alpha':0.6},
        'teacher_mlp2': {'ls': '-', 'marker': 'o', 'markersize': 2, 'capsize': 2.5, 'elinewidth': 0.8, 'alpha':0.6},
        'student_mlp2': {'ls': '--', 'marker': 'o', 'markersize': 2, 'capsize': 2.5, 'elinewidth': 0.8, 'alpha':0.6},
    }
    
    for exp_info in filtered_data.values():
        params = exp_info['params']
        data = exp_info['data']
        full_type = params['full_type']
        
        color_dict = DISEASE_PALETTE[params['disease']]
        color = color_dict[full_type]
        
        ages = sorted(map(float, data.keys()))
        p10_means = [data[a]['p10'][0] for a in ages]
        p10_stds = [data[a]['p10'][1] for a in ages]
        p11_means = [data[a]['p11'][0] for a in ages]
        p11_stds = [data[a]['p11'][1] for a in ages]

        error_multiplier = 0.0 # whether to use error bar
        
        ax1.errorbar(ages, p10_means, yerr=np.array(p10_stds)*error_multiplier,
                    color=color,
                    **style_config[full_type])
        ax2.errorbar(ages, p11_means, yerr=np.array(p11_stds)*error_multiplier,
                    color=color,
                    **style_config[full_type])

    for ax in [ax1, ax2]:
        ax.set_xlim(35, 75)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
    
    ax1.set_title('P(Positive → Negative)', fontsize=14)
    ax2.set_title('P(Positive → Positive)', fontsize=14)
    
    plt.figlegend(
        handles=create_hierarchical_legend(config),
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 1.25),
        fontsize=9,
        frameon=False
    )

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    filename = f"{'_'.join(config['diseases'])}_age{config['age_thresholds']}_es{config['es_values']}.png"
    plt.savefig(output_dir/filename, bbox_inches='tight', dpi=350)
    print(f"Visualization saved to: {output_dir/filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='age_transition_entry_65_sp1_feat_lr34.json')
    parser.add_argument('--diseases', nargs='+', default=['uc'], choices=['ad', 'ms', 'uc'])
    parser.add_argument('--model_types', nargs='+', default=['teacher', 'student'], choices=['teacher', 'student'])
    parser.add_argument('--age_thresholds', nargs='+', type=int, default=[65])
    parser.add_argument('--es_values', nargs='+', type=int, default=[20])
    parser.add_argument('--output_dir', default='plots')
    args = parser.parse_args()

    filtered_data = load_and_filter_data(args.json_file, {
        'diseases': args.diseases,
        'model_types': args.model_types,
        'age_thresholds': args.age_thresholds,
        'es_values': args.es_values
    })
    
    if not filtered_data:
        print("No such file.")
        return
    
    plot_with_error_bars(filtered_data, {
        'diseases': args.diseases,
        'age_thresholds': args.age_thresholds,
        'es_values': args.es_values,
        'output_dir': args.output_dir
    })

if __name__ == '__main__':
    main()