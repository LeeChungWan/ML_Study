import tensorflow as tf

# (1) Build graph using TF operations
# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]
# tensorflow가 사용하는 variable / tf가 자체적으로 변경시키는 값. / trainable value라고 생각해된다.
# W, b의 값을 몰라서 랜덤값으로 값이 하나인 1차원 arr로 준다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Our hypothesis XW + b
hypothesis = x_train * W + b
# cost/loss function
# t = [1., 2., 3., 4. ]
# tf.reduce_mean(t) ==> 2.5
# tf.reduce_mean(data) 은 data의 값을 평균 내주는 역할.
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

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
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        #print(step, cost, W, b)
