import os
import json
import csv
from pathlib import Path
import re

def extract_summary_data(root_dir, output_csv, disease_name=None):
    results_data = []
    root_path = Path(root_dir)

    print(f"开始在 {root_path} 中搜索 summary.json 文件...")

    for json_path in root_path.rglob('summary.json'):
        try:
            print(f"  找到: {json_path}")
            experiment_dir_name = json_path.parent.name
            model_dir_name = json_path.parent.parent.name

            run_type = "unknown"
            if "teacher" in experiment_dir_name:
                run_type = "teacher"
            elif "student" in experiment_dir_name:
                run_type = "student"

            # 从 experiment_dir_name 提取学习率信息
            lr_match = re.search(r'lr([\d.eE+-]+)', experiment_dir_name)
            learning_rate = lr_match.group(1) if lr_match else "unknown" # 提取 lr 后面的部分

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            test_auprc_str = None
            if run_type in data and 'test_auprc' in data[run_type]:
                 test_auprc_str = data[run_type]['test_auprc']

            if test_auprc_str is not None:
                try:
                    parts = test_auprc_str.split(' ± ')
                    if len(parts) == 2:
                        auprc_mean_str = parts[0]
                        auprc_std_str = parts[1]
                        auprc_mean = float(auprc_mean_str)
                        auprc_std = float(auprc_std_str)

                        # 将数据添加到结果列表，experiment 列只包含学习率，添加疾病信息
                        results_data.append({
                            'disease': disease_name,  # 添加疾病名称
                            'model_name': model_dir_name,
                            'run_type': run_type,
                            'experiment': learning_rate, # 使用提取的学习率
                            'test_auprc_mean': f"{auprc_mean:.4f}",
                            'test_auprc_std': f"{auprc_std:.4f}"
                        })
                    else:
                        print(f"    警告: 字符串 '{test_auprc_str}' 格式不符合 'mean ± std'。")

                except (ValueError, IndexError) as parse_error:
                     print(f"    警告: 无法从字符串 '{test_auprc_str}' 解析 AUPRC 均值或标准差: {parse_error}")
            else:
                print(f"    警告: 在 {json_path} 中未找到预期的 'test_auprc' 数据或 '{run_type}' 键。")

        except json.JSONDecodeError:
            print(f"    错误: 无法解析 JSON 文件 {json_path}")
        except FileNotFoundError:
            print(f"    错误: 文件未找到 {json_path}")
        except Exception as e:
            print(f"    处理文件 {json_path} 时发生未知错误: {e}")

    return results_data

def process_experiment(experiment_path, disease_list):
    """处理指定实验路径下的所有疾病数据"""
    all_results = []
    experiment_path = Path(experiment_path)
    experiment_name = experiment_path.name  # 提取实验名称
    
    print(f"处理实验: {experiment_name}")
    
    for d in disease_list:
        search_root = experiment_path / d
        print(f"处理疾病: {d}, 路径: {search_root}")
        
        # 收集每个疾病的数据
        disease_results = extract_summary_data(search_root, None, disease_name=d)
        all_results.extend(disease_results)
    
    # 将所有结果写入单个CSV文件
    output_file = f'{experiment_name}_summary_results.csv'
    print(f"\n正在将提取的数据写入 {output_file}...")
    
    if not all_results:
        print("未找到任何有效的 summary.json 文件或数据。")
        return
    
    # 更新CSV文件头以包含disease字段
    fieldnames = ['disease', 'model_name', 'run_type', 'experiment', 'test_auprc_mean', 'test_auprc_std']
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV 文件 {output_file} 写入成功！")
    except IOError as e:
        print(f"写入 CSV 文件时出错: {e}")
    except Exception as e:
        print(f"写入 CSV 时发生未知错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义疾病列表
    diseases = ['ad', 'ms', 'uc', 'af']
    
    # # 处理特定实验
    # experiment_path = '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_2'
    # process_experiment(experiment_path, diseases)
    
    experiments = [
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_5',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_11',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_12',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_13',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_14',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_15',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_16',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_17',
        '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_new1',
        '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_new2',
        '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_lc_new3',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s2',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s3',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s4',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s5',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s7',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s8',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s9',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_s10',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_gce_1',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_gce_2',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_gce_3',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_gce_4',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_mixup_1',
        # '/data3/lihan/projects/zisen/ageaware/experiments/test_new_data_mixup_2',
    ]
    for exp_path in experiments:
        process_experiment(exp_path, diseases)