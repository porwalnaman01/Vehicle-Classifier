import cv2

import os
import random
import numpy as np

train_data_path = r'D:/vehicle/train/train'
test_data_path =r'D:/vehicle/test'


#trainDataGen = ImageDataGenerator(rescale=1/255.0,width_shift_range=0.2,horizontal_flip=True,height_shift_range=0.2,zoom_range=0.2,shear_range=0.2)
#testDataGen = ImageDataGenerator(rescale=1/255.0)

categories = ['Ambulance','Bicycle','Bus','Car','Motorcycle'
              ,'Taxi','Truck','Van']
training_data = []
testing_data = []
for category in categories:
    path = train_data_path+'/'+category
    label = categories.index(category)
    for imageFile in os.listdir(path):
        try:
            image_array = cv2.imread(os.path.join(path,imageFile))
            img_array = cv2.resize(image_array,(60,60))
            img_array=img_array/255.0
            print(img_array)
            training_data.append([img_array,label])
        except:
            pass

for imgFile in os.listdir(test_data_path):
    try:
        img_array = cv2.imread(os.path.join(test_data_path,imgFile))
        img_array = cv2.resize(img_array,(60,60))
        img_array=img_array/255.0
        print(img_array)
        testing_data.append(img_array)
    except:
        pass

random.shuffle(training_data)

features= []
labels = []
for feature, label in training_data:
    features.append(feature)
    labels.append(label)
features = np.array(features).reshape(-1,60,60,3)
labels = np.array(labels)
test_data = np.array(testing_data).reshape(-1,60,60,3)
print('saving')
np.savez_compressed('Vehicles_training_features_newer.npz', features)
np.savez_compressed('Vehicles_training_labels_newer.npz',labels)
np.savez_compressed('Vehicles_testing_features_newer.npz',test_data)
