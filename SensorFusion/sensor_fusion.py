from detect import detect
import subprocess
from mlp_detector import MLP
import pandas as pd

# Load YOLOv7 model
weights = 'trained_model/best.pt'  # path to the model weights
device = 0  # automatically select a device (CPU or GPU)
img_path = 'images/bottle1.jpg'
img_size = 640

data = ['battery', 'bottle', 'cardboard', 'face_mask', 'food_leftover', 'food_peeling', 'gadget', 'glove', 'paper', 'soft_plastic', 'tetra_pack', 'tin_can']
def waste_clasify(weights, img_path):
    inference = subprocess.check_output(['python', 'detect.py', '--weights', weights, '--source', img_path], shell=True)
    # print(f'INFERENCE: {inference}')

    text = inference.decode("utf-8")
    return text

def SensorFusion():
    # initialize MLP
    mlp = MLP()
    
    # Get data from Yolov7 and sensors
    yolov7_pred = waste_clasify(weights, img_path)

    new_data = [[yolov7_pred, 1, 1]]

    new_data_df = pd.DataFrame(new_data, columns=['yolov_pred', 'capacitive_pred', 'inductive_pred'])

    # Use the trained MLP model to make predictions on the new data
    predictions = mlp.predict(new_data_df)
    print(data[predictions[0]])



SensorFusion()

  
