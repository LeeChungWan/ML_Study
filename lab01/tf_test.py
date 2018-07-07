import tensorflow as tf

# tf.constant를 이용해서 문자열이 들어간 Node를 만든다.
hi = tf.constant("Hello, TensorFlow!")

# Session을 만든 뒤, 실행해야 한다.
sess = tf.Session()

print(sess.run(hi))
print(str(sess.run(hi), encoding='utf-8'))