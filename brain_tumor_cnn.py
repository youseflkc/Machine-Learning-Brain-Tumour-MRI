import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


X_train = []
Y_train = []
X_test = []
Y_test = []
img_size = 150

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# reads in all the image files from the training set and appends them to X_train
for label in labels:
    dir = os.path.join('Training', label)
    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename))
        img = cv2.resize(img, (img_size, img_size))
        X_train.append(img)
        Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# plot the training data distribution
n, bins, patches = plt.hist(Y_train, width=0.5)
patches[0].set_facecolor('r')
patches[1].set_facecolor('g')
patches[2].set_facecolor('b')
patches[3].set_facecolor('y')
plt.title("Training Data Distribution")
plt.xlabel("Image Classification")
plt.ylabel("# of Images")
plt.show()

# split the data set into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train, Y_train, test_size=0.2)

# encodes the training image labels into numeric values
Y_train_tmp = []
for i in Y_train:
    Y_train_tmp.append(labels.index(i))
Y_train = Y_train_tmp
Y_train = tf.keras.utils.to_categorical(Y_train)

# encodes the test image labels into numeric values
Y_test_tmp = []
for i in Y_test:
    Y_test_tmp.append(labels.index(i))
Y_test = Y_test_tmp
Y_test = tf.keras.utils.to_categorical(Y_test)

# training model with 3 convolutial layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

# train the model on the training data
history = model.fit(X_train, Y_train,
                    epochs=15, validation_data=(X_test, Y_test))

# plot the results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# load test data
X_test = []
Y_test = []

for label in labels:
    dir = os.path.join('Testing', label)
    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename))
        img = cv2.resize(img, (img_size, img_size))
        X_test.append(img)
        Y_test.append(label)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# plot the testing data distribution
n, bins, patches = plt.hist(Y_test, width=0.5)
patches[0].set_facecolor('r')
patches[1].set_facecolor('g')
patches[2].set_facecolor('b')
patches[3].set_facecolor('y')
plt.title("Test Data Distribution")
plt.xlabel("Image Classification")
plt.ylabel("# of Images")
plt.show()

# encode test labels numerically
Y_test_tmp = []
for i in Y_test:
    Y_test_tmp.append(labels.index(i))
Y_test = Y_test_tmp
Y_test = tf.keras.utils.to_categorical(Y_test)

# use model to predict the test data
pred = model.predict(X_test)

# argmax gets the index of the maximum predicted outcome for each row
pred = np.argmax(pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)

# generates report for prediction results
clf_report = classification_report(
    Y_test, pred, output_dict=True, target_names=labels)

# plot classification report
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)

# plots confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(Y_test, pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
            alpha=0.7, linewidths=2)
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', y=0.92, x=0.28, alpha=0.8)

plt.show()
