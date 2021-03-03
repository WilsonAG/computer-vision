import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

iris = np.loadtxt('./data/iris.csv', delimiter=',')
X = iris[:, :-1]
Y = iris[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=20, random_state=1, max_iter=300)
model.fit(X, Y)

new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.1, 5.3, 1.9],
    [5.1, 3.1, 1.3, 0.2],
    [6.8, 3.3, 4.1, 1.1],
    [5.9, 2.9, 4.6, 1.3]
])

predictions = np.uint8(model.predict(new_flowers))
pred_labels = [classes[i] for i in predictions]

print(pred_labels)