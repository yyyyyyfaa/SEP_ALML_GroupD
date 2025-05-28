"""Source code for the shapiq_student package."""
#Create an empty class to storage train_data
#here we keep every training data, and use these when we need to predict
#so we need 2 variables self.X_train and self.y_train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class KNearestNeighbors:
    def __init__(self):
        self.X_train = None ## features of the training samples (2-dimension array)
        self.y_train = None ## lables of the trainning sample(1-d array)
#receive data from outside
    def train(self,X,y):
        self.X_train = X
        self.y_train = y
#Given one or more test points, find the nearest k training points, and use
#them to do a majority voting
    def predict(self,X,k=1):
        num_test = X.shape[0]##shape()method return a tuple to represent the dimension of array
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))# initialize dist function
        predictions = []
#As we know, (X-x_train)*(X-x_train)=X^2 + x_train^2 + -2X*x_train
        d1 = -2 * np.dot(X, self.X_train.T)
        d2 = np.sum(np.square(X), axis=1)
        d3 = np.sum(np.square(self.X_train), axis=1)
        distance = np.sqrt(d1 + d2 + d3)
        for i in range(num_test):
            #calculate every distance from test point to training point
            dists = distance[i]
            #get index of k points with smallest distances
            nearest_index = np.argsort(dists)[:k]
            #get lables
            nearest_lables = self.y_train[nearest_index]
            #majority voting
            counts = np.bincount(nearest_lables)
            prediction = np.argmax(counts)
            predictions.append(prediction)

        return np.array(predictions)
        

     def threshold_knn_shapley(self, x_test: np.ndarray, threshold: float = 1.0, epsilon: float = 1e-8) -> np.ndarray:
    from sklearn.metrics import pairwise_distances

    n_test = x_test.shape[0]
    n_train = self.x_train.shape[0]
    shap_values = np.zeros((n_test, n_train))

    model_output_test = self.model.predict(x_test)
    distances = pairwise_distances(x_test, self.x_train)

    threshold_mask = distances < threshold
    label_match_mask = (self.y_train[None, :] == model_output_test[:, None])
    contribution = threshold_mask & label_match_mask
    contribution = contribution.astype(float)

    normalization = np.sum(threshold_mask, axis=1, keepdims=True) + epsilon
    shap_values = contribution / normalization
    return shap_values

 


