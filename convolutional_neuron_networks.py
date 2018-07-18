# Building a simple CNN
#

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math

print(tf.__version__)
print()

#CNN configuration

# Conv layer #1
filter_size1 = 3 # a 5 * 5 filter
num_filters1 = 16 # 16 filters of first layer

# Conv lyaer #2
filter_size2 = 3 # a 5 * 5 filter
num_filters2 = 36 # there are 36 filters in the second conv layer

#Fully-connected layer
fc_size = 128

#Load data

from lib.mnist import MNIST
data = MNIST('/media/baiyu/A2FEE355FEE31FF1/mnist_dataset')

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))
print()

# number of pixels for each image
img_size = data.img_size

# number of pixels when a image is flattened
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays
img_shape = data.img_shape

# Number of classes, one classes for each 10 digits
num_classes = data.num_classes

num_channels = data.num_channels

# Helper-function for plotting images¶
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

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

images = data.x_test[0:9]
labels_no_one_hot = data.y_test_cls[0:9]
print(labels_no_one_hot)
print()

plot_images(images, cls_true=labels_no_one_hot)

def new_weights(shape):
    #init all variable to zeros, if you dont have
    #bn layer in your network, dont init like this
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(inputs, 
                   input_channels,
                   output_channels,
                   kernel_size,
                   stride=1):
    
    # and a filter / kernel tensor of shape 
    # [filter_height, filter_width, in_channels, out_channels], 
    shape = [kernel_size, kernel_size, input_channels, output_channels]

    # I will not be using bias here, only
    # weights
    weights = new_weights(shape=shape)

    layer = tf.nn.conv2d(input=inputs,
                         filter=weights,
                         strides=[1, stride, stride, 1],
                         padding='SAME')

    # we will use weights to plot
    return layer, weights

#flatten layer before fully connected layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()

    # the shape of the is assumed to be:
    # layer_shape == [batch_size, img_height, img_width, num_channels]
    num_features = layer_shape[1:4].num_elements()
    
    #reshape the layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

# helper function to create a new fully-connected layer
def new_fc_layer(inputs,
                 input_channels,
                 output_channels,
                 ):
    weights = new_weights(shape=[input_channels, output_channels])
    biases = new_bias(length=output_channels)

    #Calculate the layer as the matrix multiplication
    layer = tf.matmul(inputs, weights) + biases
    layer = tf.nn.relu(layer)

    return layer

# Placeholder
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_images = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_one_hot = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_no_one_hot = tf.argmax(y_one_hot, axis=1)

# Constructing newtork

# Conv1: a 3*3 filter size layer
# We use 2 3*3 filters to replace the original 5*5 filters
layer_conv1_1, weights_conv1_1 = new_conv_layer(inputs=x_images,
                                            input_channels=num_channels,
                                            output_channels=num_filters1,
                                            kernel_size=3)
layerbn1_1 = tf.layers.batch_normalization(layer_conv1_1)
layer_relu1_1 = tf.nn.relu(layerbn1_1)

#
layer_conv1_2, weights_conv1_2 = new_conv_layer(inputs=layer_relu1_1,
                                                input_channels=num_filters1,
                                                output_channels=num_filters1,
                                                kernel_size=filter_size2,
                                                stride=2)

layer_bn1_2 = tf.layers.batch_normalization(layer_conv1_2)
layer_relu1_2 = tf.nn.relu(layer_bn1_2)

#Conv layer #2
layer_conv2_1, weights_conv2_1 = new_conv_layer(inputs=layer_relu1_2,
                                                input_channels=num_filters1,
                                                output_channels=num_filters2,
                                                kernel_size=filter_size2,
                                                stride=1)
layer_bn2_1 = tf.layers.batch_normalization(layer_conv2_1)                
layer_relu2_1 = tf.nn.relu(layer_bn2_1)

layer_conv2_2, weights_conv2_2 = new_conv_layer(inputs=layer_relu2_1,
                                                input_channels=num_filters2,
                                                output_channels=num_filters2,
                                                kernel_size=filter_size2,
                                                stride=2)
layer_bn2_2 = tf.layers.batch_normalization(layer_conv2_2)
layer_relu2_2 = tf.nn.relu(layer_bn2_2)

# Flatten layer
# num_features is an int variable
layer_flat, num_features = flatten_layer(layer_relu2_2)

layer_fc1 = new_fc_layer(inputs=layer_flat,
                         input_channels=num_features,
                         output_channels=fc_size)
layer_relufc1 = tf.nn.relu(layer_fc1)
layer_dropout = tf.layers.dropout(layer_relufc1)

# fc #2
layer_fc2 = new_fc_layer(inputs=layer_dropout,
                         input_channels=fc_size,
                         output_channels=num_classes)

#classfier
y_pred_one_hot = tf.nn.softmax(layer_fc2)
y_pred_no_one_hot = tf.argmax(y_pred_one_hot, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_one_hot)
loss = tf.reduce_mean(cross_entropy)

# Optimizer:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# performance measure
correct_prediction = tf.equal(y_pred_no_one_hot, y_no_one_hot)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_one_hot: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

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

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = data.num_test

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_one_hot: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_no_one_hot, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


print()
print_test_accuracy()
print()

optimize(num_iterations=1)
print()
print_test_accuracy()
print()

optimize(num_iterations=99) # We already performed 1 iteration above.
print_test_accuracy(show_example_errors=True)


optimize(num_iterations=990) # We already performed 1 iteration above.
print_test_accuracy()


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

image1 = data.x_test[0]
plot_image(image1)

plot_conv_weights(weights=weights_conv1_1)
plot_conv_layer(layer=layer_relu1_1, image=image1)
