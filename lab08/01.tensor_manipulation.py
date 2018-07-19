import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Simple Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)  # rank
print(t.shape)  # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

# 2D Array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)  # rank
print(t.shape)  # shape

# Shape, Rank, Axis
t = tf.constant([1, 2, 3, 4])
tf.shape(t).eval()

t = tf.constant([[1, 2],
                 [3, 4]])
tf.shape(t).eval()

t = tf.constant(
    [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

# Matmul vs multiply
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
tf.matmul(matrix1, matrix2).eval()

# Watch out broadcasting
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
(matrix1 + matrix2).eval()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1 + matrix2).eval()

# Random values for variable initializations
tf.random_normal([3]).eval()
tf.random_uniform([2]).eval()
tf.random_uniform([2, 3]).eval()

# Reduce Mean/Sum
# 평균할때는 .을 찍어서 floating point 주의!
tf.reduce_mean([1, 2], axis=0).eval()

x = [[1., 2.],
     [3., 4.]]
tf.reduce_mean(x).eval()
# axis 즉, 축이 중요한 개념이다! rank 가 있을때 가장 안쪽에 있는 것이 가장 큰 axis 이고,
# 제일 바깥쪽이 0이며 제일 안쪽이 -1이 될 수도 있다.
# 축이 없으면 모두를 평균해라.
tf.reduce_mean(x, axis=0).eval()
tf.reduce_mean(x, axis=1).eval()
tf.reduce_mean(x, axis=-1).eval()

tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis=0).eval()
tf.reduce_sum(x, axis=-1).eval()
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()

# Argmax with axis
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()
tf.argmax(x, axis=1).eval()
tf.argmax(x, axis=-1).eval()

# **Reshape, squeeze, expand_dims
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)
tf.reshape(t, shape=[-1, 3]).eval()
tf.reshape(t, shape=[-1, 1, 3]).eval()

tf.squeeze([[0], [1], [2]]).eval()
tf.expand_dims([0, 1, 2], 1).eval()

# One hot
# depth 는 몇개의 원소가 있는지.
# rank 를 하나 올라가서 reshape 해줘야 한다.
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()

# casting
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=1).eval()

# Ones like and Zeros like
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x).eval()
tf.zeros_like(x).eval()

# Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)

# Transpose
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
