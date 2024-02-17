import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define paths to dataset
dataset_paths = ['paths of dataset images here',
                ]
data = []
labels = []

image_paths = []
predict_values = {
    "palm": 0,
    "l": 1,
    "fist": 2,
    "fist_moved": 3,
    "thumb": 4,
    "index": 5,
    "ok": 6,
    "palm_moved": 7,
    "c": 8,
    "down": 9
}
for dataset_path in dataset_paths:
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            try:
                file_path = os.path.join(subdir, file)
                img = cv2.imread(os.path.join(subdir, file), cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (64, 64))  # Resize to the desired input size of your model
                img = img.astype(np.float32) / 255.0  # Normalize pixel values
                data.append(img)
                labels.append(predict_values.get(subdir.split('/')[-1].split("_")[1]))  # Use the first part of the class name as label (e.g. "palm" from "palm_moved")
            except Exception as e:
                print(e)

            
data = np.array(data)
labels = np.array(labels)
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_callback = callbacks.ModelCheckpoint('hand_gesture_model.h5', monitor='val_loss', save_best_only=True)
early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_data, train_labels, epochs=10, batch_size=32,
                    validation_data=(val_data, val_labels),
                    callbacks=[checkpoint_callback, early_stopping_callback])

model.save('hand_gesture_model.h5')
