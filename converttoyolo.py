import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

train_data='./archive/images'

df= pd.read_csv('./archive/labels_train.csv')
df

df['class_id'].unique()

df.info()

labels= {0:'car',1:'truck',2:'person',3:'bicycle',4:'traffic_light'}

df.head()

width = 480
height = 300

df['x_center'] = (df['xmin'] + df['xmax']) / 2
df['y_center'] = (df['ymin'] + df['ymax']) / 2
df['width'] = (df['xmax'] - df['xmin'])
df['height'] = (df['ymax'] - df['ymin'])
df['class_id'] = df['class_id'] - 1

# normalizing the bounding box coordinates for yolo format
df["x_center"]=(df["x_center"]/width).round(3)
df["y_center"]=(df["y_center"]/height).round(3)
df["width"]=(df["width"]/width).round(3)
df["height"]=(df["height"]/height).round(3)

df_yolo=df[["frame","class_id","x_center","y_center","width","height"]]
df_yolo.head()
