#learn about basic concept in Tensorflow
#how do define a computational graph
#loss function
#optimizer

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from lib.mnist import MNIST

print(tf.__version__)
print()

data = MNIST('/media/baiyu/A2FEE355FEE31FF1/mnist_dataset')

print('size of:')
print('Training dataset {}'.format(data.num_train))
print('Validation dataset {}'.format(data.num_val))
print('Test dataset {}'.format(data.num_test))
print()


# The images are stored in one-dimensional arrays of this length
# 784
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays
# (28, 28)
img_shape = data.img_shape

# Number of classes
# 10
num_classes = data.num_classes

#one - hot encoding
print('one - hot encoding')
print(data.y_test[0:5])
print()

#class_number
print('class_number')
print(data.y_test_cls[0:5])
print()


#Helper function to polt 9 images in a 3*3 grid
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Test the helper function
images = data.x_test[0:9] #(784,) for each image
cls_true = data.y_test_cls[0:9] #get gt labels 
plot_images(images, cls_true)


# Define computational graph

# Placeholder variables
# define a Placeholder variable for the input images
# each image is a 784 length vector, None stands for
# arbitrary number of images
x = tf.placeholder(tf.float32, [None, img_size_flat])
print('image shape')
print(x.get_shape())
print()

# Same as above, the length for each label is 10
# one - hot label
y_true = tf.placeholder(tf.float32, [None, num_classes])
print('one - hot label')
print(y_true)
print()

# true label
y_true_cls = tf.placeholder(tf.float32, [None])
print('plain label')
print(y_true_cls)
print()


# Define weights
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))


# Model
logits = tf.matmul(x, weights) + biases
outputs = tf.nn.softmax(logits)
predicted_labels = tf.argmax(outputs, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# Accuracy
correct_prediction = tf.equal(y_true_cls, tf.cast(predicted_labels, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create tensorflow Session

session = tf.Session()
session.run(tf.global_variables_initializer()) #init weights and biases

# Helper functions to perform a number of optimization
# during each iteration, send a new batch of data to the
# model

batch_size = 100

def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

# Helper function to show performance
feed_dict_test = {x: data.x_test,
                  y_true: data.y_test,
                  y_true_cls: data.y_test_cls}

# Function for plotting examples of images from the test-set that have been mis-classified.

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, predicted_labels],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

# Helper-function to plot the model weights
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



# for 1 iteration:

optimize(num_iterations=1)
print_accuracy()
plot_example_errors()
plot_weights()


# run 9 iterations:
optimize(num_iterations=9)

print_accuracy()
plot_example_errors()
plot_weights()


# run anther 990 iterations
optimize(num_iterations=990)
print_accuracy()
plot_example_errors()