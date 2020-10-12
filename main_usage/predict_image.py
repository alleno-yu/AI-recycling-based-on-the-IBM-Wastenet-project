import imageio
from PIL import Image
import numpy as np
import os
import random

from googlenet_model.googlenet import create_googlenet
from config import images_path, waste_class, weights_path

model = create_googlenet(weights_path)


images = os.listdir(images_path)

random_image = random.choice(images)
random_image = os.path.join(images_path, random_image)

img = imageio.imread(random_image, pilmode='RGB')
img = np.array(Image.fromarray(img).resize((224, 224)))
img = img.transpose((2, 0, 1))

img_array = np.expand_dims(img, axis=0)

prediction = model.predict(img_array)[2][0]
argmax = np.argmax(prediction)
print(waste_class[argmax])
print("probability:{:.2f}%".format(max(prediction)*100))

im = Image.open(random_image)
im.show()



