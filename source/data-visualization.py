# visualize a subset of the training data using PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


if __name__ == "__main__":

    # load data 
    training_images = np.load("cleaned_animals_dataset/train_data_images.npy")
    training_labels = np.load("cleaned_animals_dataset/train_data_labels.npy")

    # Select a balanced subset of the training data
    num_samples_per_class = 500

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
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Animal Species')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Species Label')
    plt.show()
