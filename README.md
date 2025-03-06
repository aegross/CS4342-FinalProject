# CS4342-FinalProject
(explanation of project)

## Dataset: Animals-10 on Kaggle ([link](https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download))
(explanation of dataset)

## Machine Learning Models
### Simple Model: Softmax Regression w/ Stochastic Gradient Descent
Implementation of softmax regression was taken from the third homework from the same course, and altered to work with 
the dataset used for this project.

### Deep Model: 
The deep learning model is a convolutional neural network (CNN) designed for classifying animal images into 10 categories. It consists of three convolutional blocks with batch normalization, ReLU activation, and max pooling, followed by a fully connected classifier that outputs the predicted class labels. The model is trained using the Adam optimizer with a low learning rate and incorporates techniques like learning rate scheduling, gradient clipping, and early stopping to improve performance and generalization.

### Final Model: ???

## Run Instructions:
data-cleaning.py is designed to be run once from the source folder as the main set to retrieve the dataset from kaggle.