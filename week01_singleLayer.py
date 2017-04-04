import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 손글씨 0~9

#Placeholder
# 직접 넣어주는 데이터 / 손글씨 28x28 그림한장, 정답 (0 0 1 0 0 0 0 0 0 0) -> 0
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Variable
# 입력값 784 -> w(784x10) -> 10x1 + b -> output
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Graph
z = tf.matmul(x, W) + b
y = tf.nn.softmax(z)
# output_z -> 확률값으로(softmax) -> y
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 답하고 차이의 평균값
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step -> 학습

#Session
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))