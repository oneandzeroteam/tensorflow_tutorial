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

def summary(tensor):
    tf.summary.scalar(tensor.op.name, tensor)

summary(loss)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.5, 0.90).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer().minimize(loss)

#Session
init = tf.global_variables_initializer()

#Evaluation
pred = tf.argmax(y, 1,name='pred')
truth = tf.argmax(y_, 1, name='truth')
correct_prediction = tf.equal(pred, truth, name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
summary(accuracy)

summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter("logs", sess.graph)

for i in range(50000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100) is 0:
        acc = accuracy.eval(session=sess, feed_dict={x: batch_xs,
                                 y_: batch_ys})
        print("Epoch "+str(i)+" : " + str(acc))

        # Update the events file.
        summary_str = sess.run(summary, feed_dict={x: batch_xs,
                                 y_: batch_ys})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

acc = accuracy.eval(session=sess, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})
print("FinalAccuracy : " + str(acc))

