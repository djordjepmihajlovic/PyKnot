# import tensorflow as tf
import torch 
from torch.utils.data import random_split

# x = tf.constant([1, 4])
# y = tf.constant([2, 5])
# z = tf.constant([3, 6])

# l = tf.stack([x, y, z], axis = 1)

x1 = torch.tensor([1, 4])
x2 = torch.tensor([2, 5])
x3 = torch.tensor([3, 6])

data = torch.stack([x1, x2, x3], dim = 0)

# l1 = torch.stack([x[2], x[1], x[0]], dim = 1)

# t1 = torch.reshape(l1, (3, 3))

# m1 = torch.mean(l1, 0)

# generator1 = torch.Generator().manual_seed(42)

# xthi, xthp= random_split(x, [3, 7], generator=generator1)
print(data)
# batched_func = torch.func.vmap(lambda *x: torch.reshape(torch.stack(x, dim=1), (1, 16, 4)))
batched_func = torch.func.vmap(lambda *x: torch.cat(x, dim=0))
batched_reshape = torch.func.vmap(lambda *x: torch.reshape(data, (1, 1, 6)))
nu = batched_reshape(data)

print(nu)




