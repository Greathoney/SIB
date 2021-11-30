import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mediapipe_hands = tf.lite.Interpreter("/home/great/repos/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
mediapipe_hands.allocate_tensors()

mediapipe_hands_input_index = mediapipe_hands.get_input_details()[0]["index"]
mediapipe_hands_output_index = mediapipe_hands.get_output_details()[0]["index"]

# print(mediapipe_hands.get_input_details())
# print(mediapipe_hands.get_output_details())

mediapipe_hands_predictions = []

# find x_test
x_test = []
# x_test.append(np.random.randint(255, size=(224, 224, 3), dtype=np.int32))
x_test.append(np.random.rand(224,224,3))
plt.imshow(x_test[0])

for x_value in x_test:
    x_value_tensor = tf.convert_to_tensor([x_value], dtype = np.float32)
    mediapipe_hands.set_tensor(mediapipe_hands_input_index, x_value_tensor)
    mediapipe_hands.invoke()
    mediapipe_hands_predictions.append(mediapipe_hands.get_tensor(mediapipe_hands_output_index)[0])

print(mediapipe_hands_predictions)