import tensorflow as tf
# tensorflow의 기본 3단계 흐름
# (1) Build graph using TensorFlow operations / tf.placeholder를 이용해서 값의 자료형만 주어진 노드를 만들 수 있다.
# (2) feed data and run graph (operation) / sess.run(op, feed_dict={x:x_data})
# (3) update variables in the graph (and return values)

# placeholder를 이용해서 node를 미리 만들어 논다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shorcut for tf.add(a, b)

sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))