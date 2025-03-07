# visualize a subset of the training data in 2D and 3D using PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')

if __name__ == "__main__":

    # load data 
    training_images = np.load("cleaned_animals_dataset/train_data_images.npy")
    training_labels = np.load("cleaned_animals_dataset/train_data_labels.npy")

    # balanced subset of the training data
    num_samples_per_class = 1000

    # separate the data by class
    unique_classes = np.unique(training_labels)
    selected_indices = []

    # for each class, randomly choose a subset of the indices
    for c in unique_classes:
        c_indices = np.where(training_labels == c)[0]

        # randomly sample the indices
        if len(c_indices) > num_samples_per_class:
            c_sample = np.random.choice(c_indices, num_samples_per_class, replace=False)

        # if there are fewer than needed, take all available
        else:
            c_sample = c_indices
        selected_indices.extend(c_sample)


    # randomize the order
    np.random.permutation(selected_indices)


    # subset the data
    X_subset = training_images[selected_indices]
    y_subset = training_labels[selected_indices]


    # flatten for PCA (each image becomes a vector)
    X_flat = X_subset.reshape(X_subset.shape[0], -1)



    # apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)


    # plot the PCA results
    speciesNames = (['Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel'])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset, cmap='viridis', s=50, alpha=0.7)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Animal Species')

    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label('Species')
    cbar.set_ticklabels(speciesNames)
    cbar.set_ticks(np.arange(len(speciesNames)))
    cbar.ax.yaxis.tick_left()

    plt.show()

    

    # apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_flat)


    # 3D plot the PCA results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],c=y_subset, cmap='viridis', s=50, alpha=0.7)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Visualization of Animal Species')

    cbar = plt.colorbar(scatter, pad=0.2)
    cbar.set_label('Species')
    cbar.ax.yaxis.tick_left()
    cbar.set_ticks(np.arange(len(speciesNames)))
    cbar.set_ticklabels(speciesNames)

    plt.show()