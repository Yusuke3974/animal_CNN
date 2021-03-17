from keras.models import Sequential, Model, load_model
from tensorflow import keras
import numpy as np
from PIL import Image
import sys

classes = ["monkey","boar","bear","cat","giraffe","gorilla","horse","lion","penguin","rabbit","crow"]
num_classes = len(classes)
image_size = 224

image = Image.open(sys.argv[1])
image = image.convert('RGB')
image = image.resize((image_size, image_size))
data = np.asarray(image) / 255.0

x = []
x.append(data)
x = np.array(x)


model = load_model('./vgg16_transfer.h5')

result = model.predict([x])[0]
predicted = result.argmax()
percentage = int(result[predicted] * 100)

print(classes[predicted], percentage)