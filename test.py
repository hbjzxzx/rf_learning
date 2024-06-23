import torch
from torch.distributions import Categorical

# 假设 batched_action_prob 是一个包含概率向量的矩阵，每一行代表一个概率分布
batched_action_prob = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])

# 创建 Categorical 分布
distributions = Categorical(probs=batched_action_prob)

# 从分布中采样
samples = distributions.sample()

# 计算特定动作的对数概率
log_probs = distributions.log_prob(samples)

print("Samples:", samples)
print("Log probabilities:", log_probs)