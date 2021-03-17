from PIL import Image
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split

classes = ['monkey','boar','crow','bear','cat','giraffe','gorilla','horse','lion','penguin','rabbit']
num_classes = len(classes)
image_size = 224


x = []
y = []

for index, class_label in enumerate(classes):
    photos_dir = './images/' + class_label
    files = glob.glob(photos_dir + '/*.jpg')
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x.append(data)
        y.append(index)

x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x,y)
xy = (X_train, X_test, y_train, y_test)
np.save('./animal224.npy',xy)