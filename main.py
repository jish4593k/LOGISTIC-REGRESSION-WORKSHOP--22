import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Generating random data for training and testing
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations))

# Building a simple logistic regression model using TensorFlow and Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(simulated_separableish_features, simulated_labels, epochs=50, verbose=0)

# Model evaluation
loss, accuracy = model.evaluate(simulated_separableish_features, simulated_labels)
print(f"Accuracy from Keras: {accuracy}")

# Visualize the data
sns.set(style="darkgrid")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=simulated_separableish_features[:, 0], y=simulated_separableish_features[:, 1], hue=simulated_labels)
plt.title("Data Visualization")
plt.show()

# Create a simple Tkinter GUI to predict using the trained model
def predict():
    input_data = np.array([float(entry1.get()), float(entry2.get())]).reshape(1, 2)
    prediction = model.predict(input_data)
    result_label.config(text=f"Prediction: {prediction[0,0]:.4f}")

root = tk.Tk()
root.title("Logistic Regression Prediction")
root.geometry("300x150")

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill='both', expand=True)

label1 = ttk.Label(frame, text="Feature 1:")
label1.grid(row=0, column=0)
entry1 = ttk.Entry(frame)
entry1.grid(row=0, column=1)

label2 = ttk.Label(frame, text="Feature 2:")
label2.grid(row=1, column=0)
entry2 = ttk.Entry(frame)
entry2.grid(row=1, column=1)

predict_button = ttk.Button(frame, text="Predict", command=predict)
predict_button.grid(row=2, columnspan=2)

result_label = ttk.Label(frame, text="")
result_label.grid(row=3, columnspan=2)

root.mainloop()
