import tensorflow as tf
import numpy as np
import json


labelsValues = [
    "Extremely Weak",
    "Weak",
    "Normal",
    "Overweight",
    "Obesity",
    "Extreme Obesity"
]


sex = 0.0
height = 187/199
weight = 62/160


# Male,187,62,1 Weak


model = tf.keras.models.model_from_json(
    json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")

print(labelsValues[np.argmax(model.predict(
    tf.constant([[sex, height, weight]], dtype=tf.float32)))])
