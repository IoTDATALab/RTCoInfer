import torch as t

input=t.rand(2,2)
print(input)
a=t.max(input)
b=t.min(input).item()
print(b)
scale=(a-b)/256
Q=t.quantize_per_tensor(input,scale=scale,zero_point=b)
print(Q)