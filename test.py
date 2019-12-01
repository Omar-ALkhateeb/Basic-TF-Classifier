import tensorflow as tf
import numpy as np
import json

# loading data
with np.load("dataset.npz") as savedData:
    data_test = tf.constant(savedData['test_x'])
    labels_test = tf.constant(savedData['test_y'])

model = tf.keras.models.model_from_json(
    json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")


predictions = model.predict(data_test, batch_size=32, verbose=1)
predictions = tf.one_hot(np.argmax(predictions, 1), 6)


# comparing model results to actal results
equals = np.sum(np.all(predictions.numpy() == labels_test.numpy(), axis=1))
print("Guess accuracy: {}".format(equals/len(data_test.numpy())))
