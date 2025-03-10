import torch

def generate_cumulative_ages(original_tensor, mask, max_age=70, min_age=40):
    mask = mask.bool()
    device = original_tensor.device
    mask = mask.to(device)
    
    upper_bounds = torch.where(mask, max_age, original_tensor)
    lower_bounds = torch.where(mask, original_tensor, min_age)
    ranges = upper_bounds - lower_bounds + 1
    
    rand = torch.rand(original_tensor.shape, device=device)
    random_offsets = (rand * ranges.float()).long()
    cumulative_ages = lower_bounds + random_offsets

    return cumulative_ages


original = torch.tensor([70.0]).cuda()
mask = torch.tensor([1]).cuda()
# print(generate_cumulative_ages(original, mask))

for i in range(100000):
    new_ages = generate_cumulative_ages(original, mask)
    if new_ages[0] != torch.tensor(70.0):
        print(f"Found {i}th case")
        print(new_ages)
        break