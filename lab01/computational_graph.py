import tensorflow as tf

# (1) Build graph (tensors) using TensorFlow operations
# graph를 만들어 주는 과정.
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

# Session을 안하고 실행하면 Tensor가 나온다.
print("node1:", node1 ,"node2", node2)
print("node3:", node3)

# (2) feed data and run graph (operation)
# (3) update variables in the graph (and return values)
sess = tf.Session()
# sess.run을 통해서 그래프를 실행한다.
print("sess.run(node1, node2): ", sess.run([node1, node2])) # [] 안해주면 error.
print("sess.run(node3): ", sess.run(node3))