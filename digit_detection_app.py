import random
import os

import cv2
import numpy as np

import tensorflow as tf
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import streamlit as st


def resize_to_28px(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize_data(X_train,X_test):
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    return X_train,X_test

def apply_median_filter(img):
    return cv2.medianBlur(img, 1)

def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, X_test, y_train, y_test

def make_cnn_model(X_train, X_test, y_train, y_test):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
    cnn_model.add(tf.keras.layers.Dense(10, activation='softmax'))

    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, epochs=1)

    cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)[1]
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    cnn_f1 = f1_score(y_test, y_pred_cnn, average='weighted')

    return cnn_model, cnn_accuracy, cnn_f1

def make_fnn_model(X_train, X_test, y_train, y_test):
    fnn_model = tf.keras.models.Sequential()
    fnn_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    fnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    fnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    fnn_model.add(tf.keras.layers.Dense(10, activation='softmax'))

    fnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    fnn_model.fit(X_train, y_train, epochs=1)

    fnn_accuracy = fnn_model.evaluate(X_test, y_test)[1]
    y_pred_fnn = np.argmax(fnn_model.predict(X_test), axis=1)
    fnn_f1 = f1_score(y_test, y_pred_fnn, average='weighted')

    return fnn_model, fnn_accuracy, fnn_f1

def make_plot(cnn_model, fnn_model, cnn_accuracy, fnn_accuracy, cnn_f1, fnn_f1):
    flag = ""
    while flag.lower() != "q":
        img_num = random.randint(0, 30)
        img_path = f"samples/{img_num}.png"
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path)
                img = resize_to_28px(img)
                img = np.invert(img)
                img = apply_median_filter(img)
                img = np.array([img])

                cnn_img = img[..., np.newaxis]

                cnn_prediction = cnn_model.predict(cnn_img)
                cnn_predicted_digit = np.argmax(cnn_prediction)

                fnn_prediction = fnn_model.predict(img)
                fnn_predicted_digit = np.argmax(fnn_prediction)

                print(f"Image {img_num}:")
                print(f"  CNN predicts: {cnn_predicted_digit}")
                print(f"  FNN predicts: {fnn_predicted_digit}")

                plt.figure(figsize=(7.4,6))
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.title(f"CNN: {cnn_predicted_digit}, FNN: {fnn_predicted_digit}")
                plt.suptitle(f"CNN Accuracy: {np.round(cnn_accuracy, 2)}, FNN Accuracy: {np.round(fnn_accuracy, 2)}\n"
                             f"CNN F1-Score: {np.round(cnn_f1, 2)}, FNN F1-Score: {np.round(fnn_f1, 2)}")
                plt.show()
                flag = input("İf you want to quit press Q else continue:")
            except Exception as e:
                print(f"Some error occurred: {e}")
        else:
            print(f"Image {img_num}.png not found. Trying another image.")


def main():
    X_train,X_test,y_train,y_test = load_mnist_data()
    X_train, X_test = normalize_data(X_train, X_test)

    cnn_model, cnn_accuracy, cnn_f1 = make_cnn_model(X_train, X_test, y_train, y_test)
    fnn_model, fnn_accuracy, fnn_f1 = make_fnn_model(X_train, X_test, y_train, y_test)

    make_plot(cnn_model, fnn_model, cnn_accuracy, fnn_accuracy, cnn_f1, fnn_f1)


if __name__ == "__main__":
    main()