import numpy as np
import random


def boltzmann(lst, t):
    exps = [np.exp(i/t) for i in lst]
    sum_of_exps = sum(exps)
    softmax = [j / sum_of_exps for j in exps]
    borders = [softmax[0]]
    for i in range(1, len(softmax)):
        borders.append(softmax[i] + borders[-1])
    rand_no = random.random()
    for idx in range(len(borders)):
        if rand_no < borders[idx]:
            return idx
