import numpy as np

arr = np.array([
    [1,2,3],
    [4,5,6]
])
a = ''

row, col = arr.shape

for r in range(row):
    for c in range(col):
        a += str(arr[r][c])
    a += '\n'

print(a)