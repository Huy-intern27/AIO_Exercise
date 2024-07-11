import numpy as np  # type: ignore

# Create an array from 0 to 9
data = np.arange(0, 10, 1)

# Create a 3x3 boolean array
arr1 = np.ones((3, 3)) > 0
arr2 = np.ones((3, 3), dtype=bool)
arr3 = np.full((3, 3), fill_value=True, dtype=bool)

# Get the odd elements
arr = np.arange(0, 10)
odd_element = arr[arr % 2 == 1]

# Change the odd elements to -1
arr[arr % 2 == 1] = -1

# Reshape 1d arr to 2d arr
arr = arr.reshape(2, -1)

# Concatenate arr
data1 = np.arange(10).reshape(2, -1)
data2 = np.repeat(1, 10).reshape(2, -1)
y_arr = np.concatenate((data1, data2), axis=0)
x_arr = np.concatenate((data1, data2), axis=1)

# Repeat arr
tmp = np.arange(1, 4)
repeat_arr = np.repeat(tmp, 3)
tile_arr = np.tile(tmp, 3)

# Basic function
b = np.array([2, 6, 1, 9, 10, 3, 27])
indices = np.nonzero((b >= 5) & (b <= 10))


def max(x, y):
    if x >= y:
        return x
    return y


c = np.array([5, 7, 9, 8, 6, 4, 5])
d = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max = np.vectorize(max, otypes=[int])
e = pair_max(c, d)

max_ele = np.where(c < d, d, c)
print(max_ele)
