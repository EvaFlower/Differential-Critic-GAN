# plot data in possitive.txt

import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


def read_results(file_path):
    with open(os.path.join(file_path), 'r') as f:
        data = f.readlines()
    results = []
    for line in data:
        words = line.split()
        print(words)
        results.append(float(words[-1]))
    results = np.array(results)
    return results

file_path = 'results/tune_gan_small_digits_l1_t0_r1_fb1_flag/3/eff_pair.txt' 
pos = read_results(file_path)

print(pos.shape)

res = np.sum(pos)
print("num of eff pairs:", res)

