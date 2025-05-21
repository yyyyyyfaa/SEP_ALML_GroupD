import numpy as np
import pandas as pd
from __init__ import KNearestNeighbors
import matplotlib.pyplot as plt
df = pd.read_csv("movie_dataset.csv")
X_train = df[["fight_scenes","kiss_scenes"]].values
y_train = df["label"].values

knn = KNearestNeighbors()
knn.train(X_train, y_train)
#new movie with fight=3, kiss=6
new_movie = np.array([[3,6]])
prediction = knn.predict(new_movie,k=3)

lable_map={0:"Romance",1:"action"}
print(f"The new movie with [fight=3, kiss=6] is predicted to be: {lable_map[prediction[0]]}")

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='Romance')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='blue', label='Action')
plt.scatter(new_movie[0, 0], new_movie[0, 1], c='green', marker='*', s=200, label='New')

plt.xlabel("Fight Scenes")
plt.ylabel("Kiss Scenes")
plt.legend()
plt.show()
