# CS4342-FinalProject
Our project aims to develop a machine learning model capable of accurately classifying images of animals into ten different categories. We have developed/used three different strategies to create models for this image classification problem.

## Dataset: Animals-10 on Kaggle ([link](https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download))
This dataset contains ~26K images of ten different classes of animals. The classes are: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, and squirrel. These images are raw, which meant that we had to pre-process them before use in any of the models that we built. These images were all converted to square 48x48 RGB images, and stored in .npy format. In order to run the models yourself, it is necessary to download and clean the dataset as we did. The run instructions are placed at the bottom of the README.

## Machine Learning Models
### Simple Model: Softmax Regression w/ Stochastic Gradient Descent
Implementation of softmax regression was taken from the third homework from the same course, and altered to work with the dataset used for this project.

### Deep Model: Convolutional Neural Network (Convolutional)
The deep learning model is a convolutional neural network (CNN) designed for classifying animal images into 10 categories. It consists of three convolutional blocks with batch normalization, ReLU activation, and max pooling, followed by a fully connected classifier that outputs the predicted class labels. The model is trained using the Adam optimizer with a low learning rate and incorporates techniques like learning rate scheduling, gradient clipping, and early stopping to improve performance and generalization.

### Final Model: Augmented Pre-Trained Model
Our final model takes a pre-trained neural network and augments it to train on our data. The pre-trained model we used is the animal-classifier on Hugging Face ([link](https://huggingface.co/Falcom/animal-classifier)), which is trained to classify images into the same ten categories that our dataset contained.

## Run Instructions:
data-cleaning.py is designed to be run once from the source folder as the main set to retrieve the dataset from kaggle. Multiple directories and files will be created; the .npy files are what is referenced by each of the models. 
Simple Model: simple-model.py
Deep Model: deep-model.py
Final Model: final-model-82%.py
