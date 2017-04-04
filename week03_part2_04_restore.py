import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Placeholder
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

# Variable
W1 = tf.Variable(tf.zeros([784, 200]),name='W1')
b1 = tf.Variable(tf.zeros([200]),name='b1')
W2 = tf.Variable(tf.zeros([200, 10]),name='W2')
b2 = tf.Variable(tf.zeros([10]),name='b2')


#Graph
z1 = tf.add(tf.matmul(x, W1),b1,name='z1')
a1 = tf.nn.sigmoid(z1, name='a1')
z2 = tf.add(tf.matmul(a1, W2),b2,name='z3')
y = tf.nn.softmax(z2, name='y')

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y, name='cross_entropy')
loss = tf.reduce_mean(cross_entropy, name='loss')

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#Session
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Evaluation
pred = tf.argmax(y, 1,name='pred')
truth = tf.argmax(y_, 1, name='truth')
correct_prediction = tf.equal(pred, truth, name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

sess = tf.Session()
sess.run(init)

# Restore
ckpt = tf.train.get_checkpoint_state('logs')
saver.restore(sess, ckpt.model_checkpoint_path)

acc = accuracy.eval(session=sess, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})
print("FinalAccuracy : " + str(acc))

#train_step = tf.train.MomentumOptimizer(0.1, 0.95).minimize(cross_entropy)