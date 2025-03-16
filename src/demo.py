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

# data = np.load('data/ad/X/1000017.npy')
# nonzero_mask = data != 0
# for idx, val in enumerate(data):
#     if val != 0:
#         print(f"index: {idx}, value: {val}")

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch

X = np.random.randn(10000, 100)
y = np.random.randn(10000)
X_pred = np.random.randn(1, 100)

model_lgb = lgb.LGBMRegressor(device='gpu')
model_lgb.fit(X, y)
print(f"LightGBM: {model_lgb.predict(X_pred)}")

model_xgb = xgb.XGBRegressor(device='cuda')
model_xgb.fit(X, y)
print(f"XGBoost: {model_xgb.predict(X_pred)}")