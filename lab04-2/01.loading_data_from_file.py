import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=",", dtype=np.float32)
# 전체 n개의 row 에서 -1 인 마지막 column 을 제외한 모든 column 을 가져온다.
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data)
print("*"*60)
print(y_data)