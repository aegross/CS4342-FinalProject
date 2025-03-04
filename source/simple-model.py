import numpy as np
import matplotlib.pyplot as plt

# given an n x m x m x 3(?) matrix, transpose it (so images are in columns, not rows) and add a row of all ones
# to account for the bias terms.
def reshape_and_bias(images):

    # get shape of matrix
    images_shape = images.shape

    # reshape matrix and add a row at the bottom
    images_reshaped = np.reshape(images[:,:,:,0], (images_shape[0], images_shape[1]**2))
    images_reshaped = images_reshaped.T
    fully_reshaped = np.vstack((images_reshaped, np.ones(images_shape[0])))  # adding bias term(s), all ones

    # # making sure the reshaping was done right by plotting test image
    # one_face = np.reshape(images_reshaped[:, 0], (48, 48))  # no bias for testing
    # fig, ax = plt.subplots(1)
    # ax.imshow(one_face, cmap='viridis')  # used to do grey, viridis is nicer
    # plt.show()

    return fully_reshaped  # x_tilde

# given vectors of vectors Y (one-hot ground-truth labels) and Y_hat (percent chance guesses for each category),
# calculate the percent of correct guesses. All percentages have equal weight; the category with the highest
# likelihood to be true is taken as the guess.
# (e.g., [0.05, 0.1, 0.1, 0.25, 0.2, 0.075, 0.095, 0.03, 0.05, 0.05] --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def fPC(Y, Y_hat):

    # get the index of the highest percent in each row (the max)
    Y_hat_argmax = np.argmax(Y_hat, axis=1)

    # use those indices to convert Y_hat into a vector of one-hot vectors
    Y_hat_onehot = np.zeros(Y_hat.shape)
    Y_hat_onehot[np.arange(0, Y_hat.shape[0], 1, dtype=int), Y_hat_argmax] = 1

    # determine how many rows of each match
    num_equal = np.sum(np.all(np.equal(Y_hat_onehot, Y), axis=1))
    pc = num_equal / Y_hat.shape[0]
    return pc

# # given image data, one-hot ground truth labels, a matrix of weight vectors (with biases), and an alpha value,
# # calculate the (regularized) cross-entropy loss (negative log-likelihood).
# def fCE(X_tilde, Y, W_tilde, alpha = 1e-3):
#
#     # save the number of samples (images) as n
#     n = X_tilde.shape[1]
#
#     # make a copy of W_tilde that's the same size, but doesn't include b
#     W = W_tilde.copy()
#     W[len(W) - 1] = 0  # set last element in each column to 0 (ignoring b)
#
#     # get the softmax, Y_hat, of X_tilde and W_tilde
#     Y_hat = softmax(X_tilde, W_tilde)
#
#     # calculate the CE
#     fce = np.sum(Y * np.log(Y_hat)) * (-1.0 / n)
#
#     # add the L2 regularization term to regularize (if alpha = 0, it will do nothing)
#     fce_reg = fce + ((alpha / (2 * n)) * np.sum(W * W))
#     return fce_reg

# given image data, one-hot ground truth labels, a matrix of weight vectors (with biases), and an alpha value,
# calculate the gradient of the (regularized) cross-entropy loss for use in gradient descent.
def grad_fCE(X_tilde, Y, W_tilde, alpha = 1e-3):

    # save the number of samples (images) as n
    n = X_tilde.shape[1]

    # make a copy of W_tilde that's the same size, but doesn't include b
    W = W_tilde.copy()
    W[len(W)-1] = 0  # set last element in each column to 0 (ignoring b)

    # get the softmax, Y_hat, of X_tilde and W_tilde
    Y_hat = softmax(X_tilde, W_tilde)

    # calculate the gradient CE
    fce_gradient = (X_tilde @ (Y_hat - Y)) / n

    # add the L2 regularization term to regularize (if alpha = 0, it will do nothing)
    # gradient of L2 = alpha/n * w
    fce_gradient_reg = fce_gradient + ((alpha/n) * W)
    return fce_gradient_reg

# helper function for softmax_regression() that calculates the softmax given X_tilde and W_tilde.
def softmax(X_tilde, W_tilde):
    # calculate z ("pre-activation scores") - note that Z contains row vectors, not column vectors!
    Z = X_tilde.T @ W_tilde  # Z = XTW

    # exponentiate the values of Z
    Z_exp = np.exp(Z)

    # calculate predictions, yhat, by dividing each row by the sum of the row (elementwise)
    Y_hat = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    return Y_hat

# Given training data, learning rate epsilon, batch size, and regularization strength alpha, conduct stochastic
# gradient descent (SGD) to optimize the weight matrix W_tilde (785x10). Then return W_tilde.
def softmax_regression(training_images, training_labels, epsilon, batch_size, num_epochs, alpha):

    # re-labeling to make my life easier
    X_tilde = training_images
    Y = training_labels
    n = X_tilde.shape[1]

    # generate small random numbers for the starting matrix of weight vectors, including bias terms
    W_tilde = np.random.randn(X_tilde.shape[0], training_labels.shape[1]) * 1.0e-5

    # perform stochastic gradient descent on the set, randomized each epoch
    num_batches = int(n/batch_size)

    for e in range(0, num_epochs):
        # this is reset each epoch
        subset_start_index = 0

        # anneal the learning rate as the epochs go on
        if e == int(num_epochs/4): epsilon = epsilon / 10  # once around 1/4
        if e == int(num_epochs/2): epsilon = epsilon / 10  # another time around 1/2

        # randomize the order of the training data (randomize columns only) - and make sure the labels are randomized too
        random_indices = np.random.permutation(X_tilde.shape[1])

        # get random column indices, keep the rows as-is
        X_tilde_random = X_tilde[:, random_indices]
        # put the ROWS in the same order, since these vectors are horizontal
        Y_random = Y[random_indices, :]

        # go through all the batches in the dataset, covering everything
        for r in range(0, num_batches):
            # these will be updated every batch
            X_rand_subset = X_tilde_random[:, subset_start_index:(subset_start_index + batch_size):1]
            Y_rand_subset = Y_random[subset_start_index:(subset_start_index + batch_size):1, :]

            # calculate the gradient of the SUBSET (with all W) and adjust the weights
            W_tilde = W_tilde - (epsilon * grad_fCE(X_rand_subset, Y_rand_subset, W_tilde, alpha))

            # make sure to update the starting index to "move forward" in the list of subsets
            subset_start_index += batch_size

    # return the learned weight vector matrix
    return W_tilde

if __name__ == "__main__":

    # load data (note: the image .npy files were 4D, but the last axis is redundant; components are equal)
    training_images = np.load("cleaned_animals_dataset/train_data_images.npy")  # shape: (20938, 48, 48, 3)
    training_labels = np.load("cleaned_animals_dataset/train_data_labels.npy")  # (5241, 48, 48, 3)
    testing_images = np.load("cleaned_animals_dataset/test_data_images.npy")    # (20938,)
    testing_labels = np.load("cleaned_animals_dataset/test_data_labels.npy")    # (5241,)

    print(f"training_images shape: {training_images.shape}, training_labels shape: {training_labels.shape}")
    print(f"testing_images shape: {testing_images.shape}, testing_labels shape: {testing_labels.shape}")

    # append a constant 1 term to each example (columns) to correspond to the bias terms
    training_images_fixed = reshape_and_bias(training_images)
    testing_images_fixed = reshape_and_bias(testing_images)

    # change from 0-9 labels to "one-hot" binary vector labels. For instance,
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    onehot_training_labels = np.eye(10)[training_labels]  # 10 since values are from [0, 9]
    onehot_testing_labels = np.eye(10)[testing_labels]

    # train the model
    hyp = (1e-3, 64, 512, 1e-3) # epsilon, batch_size, num_epochs, alpha
    W_tilde = softmax_regression(training_images_fixed, onehot_training_labels, hyp[0], hyp[1], hyp[2], hyp[3])

    # get the softmax yhat value using the testing data and the trained weights
    Y_hat_training = softmax(training_images_fixed, W_tilde)
    Y_hat_testing = softmax(testing_images_fixed, W_tilde)

    # get the percent correct and print it out
    training_pc = fPC(onehot_training_labels, Y_hat_training)
    testing_pc = fPC(onehot_testing_labels, Y_hat_testing)
    print(f"Training fPC = {training_pc} ({training_pc * 100}%)")
    print(f"Testing fPC = {testing_pc} ({testing_pc * 100}%)\n")
