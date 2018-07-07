import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# shape=[None]이면 값이 몇개 들어와도 상관 없다.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW + b
hypothesis = X * W + b
# cost/loss function
# t = [1., 2., 3., 4. ]
# tf.reduce_mean(t) ==> 2.5
# tf.reduce_mean(data) 은 data의 값을 평균 내주는 역할.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# (2), (3) Run/update graph and get results
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph
# tf.Variable을 사용하면 반드시 global_variables_initializer()을 해줘야 한다.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: [1, 2, 3, 4, 5],
                                                    Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis,
               feed_dict={X: [1.5, 3.5]}))
