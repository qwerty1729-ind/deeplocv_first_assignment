# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Task 1 - Data Loading and Preprocessing
# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0, 1, 2)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Task 2 - Neural Network Construction
# Build the neural network
model = Sequential()
# Input Layer and Hidden Layer with 8 neurons and ReLU activation
model.add(Dense(8, input_dim=4, activation='relu'))
# Output Layer with 3 neurons (softmax activation for multi-class classification)
model.add(Dense(3, activation='softmax'))

# Task 3 - Model Compilation and Training
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 100 epochs with a batch size of 5
history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

# Task 4 - Model Evaluation
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

