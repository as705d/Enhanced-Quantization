import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

model = torch.load("C:\\Users\\dudal\\Desktop\\Model\\res110\\model_best.pth.tar")

#8layer
parameter = []
alpha_list = []
Layer = []
number = 1

for i in model['state_dict']:
    if "wgt_alpha" in i:
        alpha = model['state_dict'][i]
        alpha_list.append(np.round(alpha.tolist(),3))
        Layer.append(number)
        number += 1

d = len(Layer)

plt.figure(figsize=(6,6))
#plt.scatter(Layer, alpha_list, color="red")
plt.plot(Layer, alpha_list, color="red", label="Clipping value")
plt.xticks(np.arange(1, d, 10))
plt.yticks(np.arange(-5000000, 5000000, 1))
plt.xlabel('Layer')
plt.legend()
#plt.grid()
plt.show()
