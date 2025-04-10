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

import pickle
import pandas as pd
file_path = 'data/ad_new/gene2snps.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)
df = pd.DataFrame.from_dict(data, orient='index')
df.to_csv('gene2snps_1.csv', index=True)

# import pickle
# import numpy as np
# with open("data/ad_new/gene2snps.pkl", "rb") as f:
#     gene2snps = pickle.load(f)
# print(f"gene2snps['SRSF8']: {min(gene2snps['SRSF8'])}-{max(gene2snps['SRSF8'])}")
# print(f"gene2snps['MMP12']: {min(gene2snps['MMP12'])}-{max(gene2snps['MMP12'])}")
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