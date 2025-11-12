import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/home/sagor/Projects/YOLO/runs/pose/train/weights/best_saved_model/best_float32.tflite")
interpreter.allocate_tensors()

output_details = interpreter.get_output_details()
input_details = interpreter.get_input_details()

print("Input:", input_details)
print("Output:", output_details)
