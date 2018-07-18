#Keras API practice

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math


from tensorflow.python import keras


print(tf.__version__)
print()


from lib.mnist import MNIST
data = MNIST('/media/baiyu/A2FEE355FEE31FF1/mnist_dataset')

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))
print()

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = data.img_shape_full

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
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

# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
label_gt_no_one_hot = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=label_gt_no_one_hot)

def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.y_test_cls)

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

# Start construction of the Keras Sequential model
model = keras.models.Sequential()

model.add(keras.layers.InputLayer(input_shape=(img_size_flat,)))
model.add(keras.layers.Reshape(img_shape_full))

#Conv1_1
model.add(keras.layers.Conv2D(kernel_size=3, 
                              strides=1, 
                              filters=16, 
                              padding='same', 
                              kernel_initializer='he_normal'))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

#Conv1_2
model.add(keras.layers.Conv2D(kernel_size=3,
                              strides=2,
                              filters=16,
                              padding='same',
                              kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

#Conv2_1
model.add(keras.layers.Conv2D(kernel_size=3,
                              strides=1,
                              filters=36,
                              padding='same',
                              kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

#Conv2_2
model.add(keras.layers.Conv2D(kernel_size=3,
                              strides=1,
                              filters=36,
                              padding='same',
                              kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

#Flatten layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))

#FC layer1
model.add(keras.layers.Dense(num_classes, activation='softmax'))

#optimizer
optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Training
model.fit(x=data.x_train,
          y=data.y_train,
          epochs=1, batch_size=128)

result = model.evaluate(x=data.x_test,
                        y=data.y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


y_pred = model.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)


##Functional Model
inputs = keras.layers.Input(shape=(img_size_flat,))
net = inputs

net = keras.layers.Reshape(img_shape_full)(net)

#conv1_1
net = keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, 
                          padding='same')(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation('relu')(net)

#conv1_2
net = keras.layers.Conv2D(kernel_size=3, strides=2, filters=16,
                          padding='same')(net)
net = keras.layers.BatchNormalization()(net)                        
net = keras.layers.Activation('relu')(net)

#conv2_1
net = keras.layers.Conv2D(kernel_size=3, strides=1, filters=36,
                          padding='same')(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation('relu')(net)

#conv2_2
net = keras.layers.Conv2D(kernel_size=3, strides=2, filters=36,
                          padding='same')(net)
net = keras.layers.BatchNormalization()(net)
net = keras.layers.Activation('relu')(net)

#Flatten
net = keras.layers.Flatten()(net)

#first Fc1
net = keras.layers.Dense(128)(net)
net = keras.layers.Activation('relu')(net)
net = keras.layers.Dropout(0.5)(net)

#first Fc2
net = keras.layers.Dense(num_classes, activation='softmax')(net)
outputs = net

model2 = keras.models.Model(inputs=inputs, outputs=outputs)

model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

#Training
model2.fit(x=data.x_train,
           y=data.y_train,
           epochs=1, 
           batch_size=128)

result = model2.evaluate(x=data.x_test,
                         y=data.y_test)

for name, value in zip(model2.metrics_names, result):
    print(name, value)

print()
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
print()

y_pred = model2.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)


# Save and load model

path_model = 'model.keras'
model2.save(path_model)
del model2

from tensorflow.python.keras.models import load_model
model3 = load_model(path_model)
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
y_pred = model3.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images=images,
            cls_pred=cls_pred,
            cls_true=cls_true)

# Visualize layer and ouputs:

def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

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
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get Layers
model3.summary()

layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
weights_conv1 = layer_conv1.get_weights()[0] #get weights, not bias
print(weights_conv1.shape)
print()

plot_conv_weights(weights=weights_conv1, input_channel=0)


from tensorflow.python.keras import backend as K



output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


image1 = data.x_test[0]
plot_image(image1)
layer_output1 = output_conv1([[image1]])[0]

print(layer_output1.shape)
print()

def plot_conv_output(values):
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
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_conv_output(values=layer_output1)