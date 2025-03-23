# import torch

# def generate_cumulative_ages(original_tensor, mask, max_age=70, min_age=40):
#     mask = mask.bool()
#     device = original_tensor.device
#     mask = mask.to(device)
    
#     upper_bounds = torch.where(mask, max_age, original_tensor)
#     lower_bounds = torch.where(mask, original_tensor, min_age)
#     ranges = upper_bounds - lower_bounds + 1
    
#     rand = torch.rand(original_tensor.shape, device=device)
#     random_offsets = (rand * ranges.float()).long()
#     cumulative_ages = lower_bounds + random_offsets

#     return cumulative_ages


# original = torch.tensor([70.0]).cuda()
# mask = torch.tensor([1]).cuda()
# # print(generate_cumulative_ages(original, mask))

# for i in range(100000):
#     new_ages = generate_cumulative_ages(original, mask)
#     if new_ages[0] != torch.tensor(70.0):
#         print(f"Found {i}th case")
#         print(new_ages)
#         break

# import numpy as np
# for disease in ['ad', 'ad_new']:
#     data = np.load(f'data/{disease}/X/1000017.npy')
#     print(data.shape)
#     print(np.unique(data))
#     print(len(np.where(data == 0)[0]))
#     print(len(np.where(data == 1)[0]))
#     print(len(np.where(data == 2)[0]))
#     print(len(np.where(data == 3)[0]))

# for idx, val in enumerate(data):
#     if val != 0:
#         print(f"index: {idx}, value: {val}")

# import numpy as np
# import xgboost as xgb
# import lightgbm as lgb
# import torch

# X = np.random.randn(10000, 100)
# y = np.random.randn(10000)
# X_pred = np.random.randn(1, 100)

# model_lgb = lgb.LGBMRegressor(device='gpu')
# model_lgb.fit(X, y)
# print(f"LightGBM: {model_lgb.predict(X_pred)}")

# model_xgb = xgb.XGBRegressor(device='cuda')
# model_xgb.fit(X, y)
# print(f"XGBoost: {model_xgb.predict(X_pred)}")

# import pickle
# import pandas as pd
# file_path = 'data/ad_new/gene2snps.pkl'
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
# df = pd.DataFrame.from_dict(data, orient='index').T

# # 保存为 CSV 文件
# df.to_csv('gene2snps.csv', index=False)

# import pickle
# import numpy as np
# # 读取Pickle文件
# with open("data/ad_new/gene2snps.pkl", "rb") as f:
#     gene2snps = pickle.load(f)
# print(f"type(gene2snps['SRSF8']): {type(gene2snps['SRSF8'])}")
# for key, value in gene2snps.items():
#     print(f"[Key] Gene: {key}")
#     print(f"[Value] SNP list: {value[:5]}, ...\nLength: {len(value)}")
#     print("-------------")


# import pandas as pd
# df = pd.read_csv("gene2snps.csv")
# print(f"df.head(): \n{df.head()}")
# valid_num = len(df)-df.isnull().sum()
# total_num = 0
# for key, value in valid_num.items():
#     total_num += value
# print(f"valid_num: \n{valid_num}\n total number: {total_num}")
# snp_num = len(df)-df.isnull().sum()
# print(f"snp_num: \n{snp_num.sort_values(ascending=False)}")
# print(f"df['SRSF8']: \n{df['SRSF8']}")
# print(f"df['MMP12']: \n{df['MMP12']}")
# for i in [0.0, 1.0, 2.0, 3.0]:
#     target_value = i
#     columns_with_value = df.columns[df.apply(lambda col: target_value in col.values)]
#     print(f"number of columns with {target_value}: {len(columns_with_value)}")

# import numpy as np

# def compare_arrays(A, B):
#     """
#     对比两个一维 NumPy 数组的差异，返回：
#     - A 中存在但 B 中不存在的元素及其索引
#     - B 中存在但 A 中不存在的元素及其索引
#     - 两个数组的交集元素及其在各自数组中的索引
    
#     Args:
#         A (np.ndarray): 第一个数组（形状 (n,)）
#         B (np.ndarray): 第二个数组（形状 (m,)）
    
#     Returns:
#         dict: 包含差异信息的字典
#     """
#     # 1. A 中存在但 B 中不存在的元素及其索引
#     mask_A_not_in_B = np.isin(A, B, invert=True)
#     a_values = A[mask_A_not_in_B]
#     a_indices = np.where(mask_A_not_in_B)[0]
#     # 2. B 中存在但 A 中不存在的元素及其索引
#     mask_B_not_in_A = np.isin(B, A, invert=True)
#     b_values = B[mask_B_not_in_A]
#     b_indices = np.where(mask_B_not_in_A)[0]
#     # 3. 两个数组的交集元素及其索引
#     common_values = np.intersect1d(A, B)
#     common_A_indices = []
#     common_B_indices = []
#     for val in common_values:
#         a_indices_val = np.where(A == val)[0]
#         b_indices_val = np.where(B == val)[0]
#         common_A_indices.append(a_indices_val)
#         common_B_indices.append(b_indices_val)
#     return {
#         'A_not_in_B': {
#             'values': a_values,
#             'indices': a_indices
#         },
#         'B_not_in_A': {
#             'values': b_values,
#             'indices': b_indices
#         },
#         'common_elements': {
#             'values': common_values,
#             'indices_in_A': common_A_indices,
#             'indices_in_B': common_B_indices
#         }
#     }


# A = np.load('data/ad_new/X/1000017.npy').astype(float)
# B = np.load('data/ad/X/1000017.npy').astype(float)
# A = np.pad(A, (0, len(B) - len(A)), mode='constant', constant_values=np.nan)
# combined = np.column_stack((A, B))
# print(f"A.shape: {A.shape}, B.shape: {B.shape}")
# np.savetxt('data/comparison.csv', combined, delimiter=',', 
#            header='A,B', fmt='%.2f', comments='')
# result = compare_arrays(A, B)
    
    # print("A 中存在但 B 中不存在的元素及索引：")
    # print("Values:", result['A_not_in_B']['values'])
    # print("Indices in A:", result['A_not_in_B']['indices'])
    
    # print("\nB 中存在但 A 中不存在的元素及索引：")
    # print("Values:", result['B_not_in_A']['values'])
    # print("Indices in B:", result['B_not_in_A']['indices'])
    
    # print("\n两个数组的共同元素及索引：")
    # for i, val in enumerate(result['common_elements']['values']):
    #     print(f"Value {val}:")
    #     print(f"Indices in A: {result['common_elements']['indices_in_A'][i]}")
    #     print(f"Indices in B: {result['common_elements']['indices_in_B'][i]}")

# import torch
# from torch import nn
# import numpy as np
# A = torch.randn(2, 2, 5)
# A = A.view(2, 10)
# B = np.random.randn(2, 10)
# B = torch.from_numpy(B).view(2, 10)
# A_dropout = nn.Dropout(0.5)(A)
# print(f"A: {A}")
# print(f"A_dropout: {A_dropout}")