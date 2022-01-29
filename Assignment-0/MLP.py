from matplotlib import pyplot as plt
import numpy as np
import pickle


def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels).
    :param inputs_file_path: file path for ONE input batch, something like
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    # TODO: Load inputs and labels
    file = open(inputs_file_path, 'rb')
    data_dictionary = pickle.load(file, encoding='bytes')
    # TODO: Normalize inputs
    data_csv = np.array(data_dictionary[b'data'] / 255.0)
    data_labels = np.array(data_dictionary[b'labels'])
    one_hot_data_labels = one_hot_encoding(data_labels)
    return data_csv, one_hot_data_labels


def one_hot_encoding(output_list):
    one_hot_vector = np.zeros((len(output_list), 10), int)
    for index in range(len(output_list)):
        one_hot_vector[index][output_list[index]] = 1
    return one_hot_vector


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = None  # Size of image vectors
        self.num_classes = 10  # Number of classes/possible labels
        self.batch_size = 50
        self.learning_rate = 0.5

        # TODO: Initialize weights and biases
        self.W = np.zeros((10, 3072))
        self.b = np.zeros((10, 1))

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        Z1 = np.dot(self.W, inputs.T) + self.b
        A1 = softmax(Z1)
        return A1.T

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step).
        :param probabilities: matrix that contains the probabilities
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        return np.sum(np.nan_to_num(-labels * np.log(probabilities)))

    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        number_of_examples = inputs.shape[0]
        dZ1 = probabilities - labels
        gradW = (1 / number_of_examples) * np.dot(dZ1.T, inputs)
        gradB = (1 / number_of_examples) * np.sum(dZ1.T, axis=1, keepdims=True)
        return gradW, gradB

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        count = 0
        for (x, y) in zip(probabilities, labels):
            if np.argmax(x) == np.argmax(y):
                count += 1
        return count / labels.shape[0]

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W -= (self.learning_rate / float(self.batch_size) * gradW)
        self.b -= (self.learning_rate / float(self.batch_size) * gradB)


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    epochs = 1
    total_cross_entropy_loss = 0
    if len(train_inputs) < model.batch_size:
        model.batch_size = len(train_inputs)

    for i in range(epochs):
        mini_batches_data = [train_inputs[index:index + model.batch_size] for index in
                             range(0, len(train_inputs), model.batch_size)]
        mini_batches_label = [train_labels[index:index + model.batch_size] for index in
                              range(0, len(train_labels), model.batch_size)]

        for mini_batch_data, mini_batch_label in zip(mini_batches_data, mini_batches_label):
            probabilities = model.forward(mini_batch_data)
            predicted_loss = model.loss(probabilities, mini_batch_label) / len(mini_batch_label)
            gradW, gradB = model.compute_gradients(mini_batch_data, probabilities, mini_batch_label)
            model.gradient_descent(gradW, gradB)
            total_cross_entropy_loss += predicted_loss
        total_cross_entropy_loss /= len(mini_batches_data)
        # if i % 5 == 0:
        #     print('Cost of last batch after iteration {0}: {1}'.format(i, total_cross_entropy_loss))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    output = model.forward(test_inputs)

    # TODO: Return accuracy across testing set
    return model.accuracy(output, test_labels)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses) + 1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses, color='r')
    plt.draw()
    plt.pause(0.001)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main():
    '''
    Read in CIFAR10 data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch.
    :return: None
    '''

    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    data_batch_1, labels_batch_1 = get_data('cifar-10-batches-py/data_batch_1')
    data_batch_2, labels_batch_2 = get_data('cifar-10-batches-py/data_batch_2')
    data_batch_3, labels_batch_3 = get_data('cifar-10-batches-py/data_batch_3')
    data_batch_4, labels_batch_4 = get_data('cifar-10-batches-py/data_batch_4')
    data_batch_5, labels_batch_5 = get_data('cifar-10-batches-py/data_batch_5')
    test_batch, test_labels = get_data('cifar-10-batches-py/test_batch')

    train_data = np.concatenate((data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5), axis=0)
    train_labels = np.concatenate((labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5),
                                  axis=0)


    # train_data, train_labels, validation_data, validation_labels = train_data[:49000], train_labels[:49000], train_data[
    #                                                                                                          49000:], train_labels[
    #                                                                                                                   49000:]
    # Check data stratification
    # count_list = {0: 0, 1: 0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    # for label in validation_labels:
    #     count_list[np.argmax(label)] += 1
    # print(count_list)

    # print('Step 1 - Data concatenated')

    # TODO: Create Model
    model = Model()
    # print('Step 2 - Model Initialized and Created')

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_data, train_labels)
    # print('Step 3 - Model Trained')
    # Validation accuracy - Hyperparameter tuning
    # validation_accuracy = test(model, validation_data, validation_labels)
    # print('Step 4 - Validation accuracy of the model: {0:.4f}'.format(validation_accuracy))

    # TODO: Test the accuracy by calling test() after running train()
    accuracy = test(model, test_batch, test_labels)
    # print('Step 5 - Model Tested with accuracy: {0:.4f}'.format(accuracy))
    print('Model accuracy: {0:.4f}'.format(accuracy))

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    probabilities = model.forward(test_batch[:10])
    one_hot_decoded = list()
    for label in test_labels[:10]:
        one_hot_decoded.append(np.argmax(label))
    visualize_results(test_batch[:10], probabilities, one_hot_decoded)
    # print('Step 6 - Visualized Data')

if __name__ == '__main__':
    main()
