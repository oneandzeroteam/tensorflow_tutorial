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
W1_conv1 = tf.Variable(tf.truncated_normal([3,3,1,32],stddev=0.1))
b1_conv1 = tf.Variable(tf.truncated_normal([32],stddev=0.1))
W2_conv2 = tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.1))
b2_conv2 = tf.Variable(tf.truncated_normal([64],stddev=0.1))
W3_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 200],stddev=0.1))
b3_fc1 = tf.Variable(tf.truncated_normal([200],stddev=0.1))
W4_fc2 = tf.Variable(tf.truncated_normal([200, 10],stddev=0.1))
b4_fc2 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

#Graph
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.conv2d(x_image, W1_conv1, strides=[1, 1, 1, 1], padding='SAME')
a_conv1 = tf.nn.sigmoid(h_conv1)
a_pool1 = tf.nn.max_pool(a_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
h_conv2 = tf.nn.conv2d(a_pool1, W2_conv2, strides=[1, 1, 1, 1], padding='SAME')
a_conv2 = tf.nn.sigmoid(h_conv2)
a_pool2 = tf.nn.max_pool(a_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

a_pool2_flat = tf.reshape(a_pool2, [-1, 7*7*64])
h_fc1 = tf.matmul(a_pool2_flat, W3_fc1) + b3_fc1
a_fc1 = tf.nn.sigmoid(h_fc1)
h_fc2 = tf.matmul(a_fc1, W4_fc2) + b4_fc2
y = tf.nn.softmax(h_fc2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer().minimize(loss)
# train_step -> 학습

#Session
init = tf.global_variables_initializer()

#Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(init)

for i in range(50000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    i_loss,_ = sess.run([loss,train_step], feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100) is 0:
        acc = accuracy.eval(session=sess, feed_dict={x: mnist.test.images,
                                 y_: mnist.test.labels})
        print("Epoch "+str(i)+" Acc : " + str(acc) + ", Loss : " + str(i_loss))


print("FinalAccuracy : " + accuracy.eval(session=sess, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))