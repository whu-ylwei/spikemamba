import numpy as np

# ?? .npy ??
data = np.load("DENSE/train/train_sequence_00_town01/spike/r128/spike_0000000001.npy")

print(type(data))   # ??? <class 'numpy.ndarray'>
print(data.shape)   # ??????
print(data.dtype)   

print(data)      