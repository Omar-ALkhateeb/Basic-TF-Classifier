import tensorflow as tf
import numpy as np
from random import randint
import os
# from tensorflow.compat.v1 import enable_eager_execution
import json

colors = None
labels = None
data_size = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# enable_eager_execution()
np.random.seed(0)

with np.load("processedData.npz") as savedData:
    data = np.array(savedData['data'], dtype=np.float32)
    # one hot encoding
    labels = np.array([[1.0 if j == i else 0.0 for j in range(6)]
                       for i in savedData['labels']])
    data_size = len(savedData['data'])

# dividing data to 80% training anf 20% validation
train_size = int(data_size*0.8)
test_size = validation_size = int((data_size - train_size)/2)


indexes = [randint(0, data_size-1) for i in range(train_size)]

data_train = tf.constant([data[i] for i in indexes])
labels_train = tf.constant([labels[i] for i in indexes])

test_indexes = []

for i in range(0, data_size):
    if not (i in indexes):
        test_indexes.append(i)

test_indexes = [test_indexes[randint(0, test_size-1)]
                for i in range(test_size)]
data_test = tf.constant([data[i] for i in test_indexes])
labels_test = tf.constant([labels[i] for i in test_indexes])
validation_indexes = []
for i in range(0, data_size):
    if not (i in test_indexes) and not (i in indexes):
        validation_indexes.append(i)
validation_indexes = [validation_indexes[randint(
    0, validation_size-1)] for i in range(validation_size)]
data_validation = tf.constant([data[i] for i in validation_indexes])
labels_validation = tf.constant([labels[i] for i in validation_indexes])

# saving proccesed data
np.savez_compressed("dataset", train_x=data_train.numpy(), train_y=labels_train.numpy(), test_x=data_test.numpy(),
                    test_y=labels_test.numpy(), validation_x=data_validation.numpy(), validation_y=labels_validation.numpy())

# creating model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(3,), activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

# compiling
model.compile(optimizer=tf.optimizers.Adam(0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
print("Training:")
model.fit(data_train, labels_train, epochs=90, batch_size=20)
print("Training ended. Validating:")
model.fit(data_validation, labels_validation, epochs=80, batch_size=20)
json.dump({'model': model.to_json()}, open("model.json", "w"))
model.save_weights("model_weights.h5")
