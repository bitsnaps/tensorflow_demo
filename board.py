# build graph computation (a set of nodes, where each node represet a math operation)
# each node takes a tensor input and a tensor out
# a tensor is how a data is represented in multidimentional arrays of numbers
# tensor flows between operations hence the name Tensorflow

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('tmp/', one_hot=True)

# set parameters
learning_rate = 0.01  # how fast we're gonna update weights
training_iteration = 30
batch_size = 100
display_step = 2

# building placeholder operations
x = tf.placeholder("float", [None, 784]) # mnist data image of shpae (28*28)
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create the model

# set model weights
w = tf.Variable(tf.zeros([784, 10]))
# set mode biases
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
  # Construct a linear model
  model = tf.nn.softmax(tf.matmul(x, w) + b) # softmax

# Add summary ops to collect data
w_h = tf.summary.histogram("weights", w)
b_h = tf.summary.histogram("biases", b)

# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope: # cost_function helps us to reduce error during training
  # Minimize error using cross entropy
  # Cross entropy
  cost_function = -tf.reduce_sum(y*tf.log(model))
  # Create a summary to monitor the cost function
  tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
  # Gradient descent
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# initiazing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()


# merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  # set the logs writer to the folder /tmp/tensorflow_logs
#  summary_writer = tf.train.SummaryWriter('tmp/logs', graph_def=sess.graph_def)
  summary_writer = tf.summary.FileWriter('tmp/logs', graph=sess.graph)
  
  # Training cycle
  for iteration in range(training_iteration):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      # fit training using batch data
      sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
      # compute the average loss
      avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
      # write logs for each iteration
      summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
      summary_writer.add_summary(summary_str, iteration*total_batch + i)
  # display logs per iteration step
  if iteration % display_step == 0:
    print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

  # test the model
  predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
  # calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
  print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
