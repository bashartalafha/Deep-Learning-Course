import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True, validation_size=0)

first_image = mnist.train.images[4]
first_image = first_image*255
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

#import pdb; pdb.set_trace()
learning_rate = 0.06
training_iteration = 25
batch_size = 100
tf.set_random_seed(1)


# Input 
x = tf.placeholder("float", [None, 784],name="Pixles")
y = tf.placeholder("float", [None, 10],name="TrueLabels")

# Model weights
# training means computing those variables
with tf.name_scope("Hidden_Layer1") as scope:
    W1 = tf.Variable(initial_value=tf.random_normal([784,300],stddev=0.008) ,name="W1" )
    B1 = tf.Variable(initial_value=tf.zeros([300]),name="B1")
    Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)

with tf.name_scope("Hidden_Layer2") as scope:
    W2 = tf.Variable(initial_value=tf.random_normal([300, 300],stddev=0.008) ,name="W2" )
    B2 = tf.Variable(initial_value=tf.zeros([300]),name="B2")
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)

with tf.name_scope("Output_Layer") as scope:
    W3=tf.Variable(initial_value=tf.random_normal([300,10],stddev=0.05 )  ,name="W2" )
    B3 = tf.Variable(initial_value=tf.zeros([10]), name="B2")
    y_pred = tf.nn.softmax(tf.matmul(Y2, W3) + B3)

with tf.name_scope("Cost") as scope:
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
#    cost_function = -tf.reduce_sum(y*tf.log(y_pred))
    tf.summary.scalar("cost_function", cost_function)

# Optimizer
with tf.name_scope("optimizer_GD") as scope:
    # Gradient descent: minimize cost function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

# Launch graph 
with tf.Session() as sess:
    sess.run(init)
    

    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #import pdb;pdb.set_trace()
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss OPTIONAL
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            
            
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning Completed!")

    predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
   
    
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
