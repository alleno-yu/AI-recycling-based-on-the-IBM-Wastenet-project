import imageio
from PIL import Image
import numpy as np
import os
import random

from googlenet_model.googlenet import create_googlenet
from main_usage.config import images_path, waste_class, weights_path

# create the modified googlenet model
model = create_googlenet(weights_path)

# random select one image form the folder
images = os.listdir(images_path)
random_image = random.choice(images)
random_image = os.path.join(images_path, random_image)

# resize and transpose the selected image
img = imageio.imread(random_image, pilmode='RGB')
img = np.array(Image.fromarray(img).resize((224, 224)))
img = img.transpose((2, 0, 1))

# expand the image array
img_array = np.expand_dims(img, axis=0)

# use the model to predict image
prediction = model.predict(img_array)[2][0]
argmax = np.argmax(prediction)

# show waste prediction class and probability
print(waste_class[argmax])
print("probability:{:.2f}%".format(max(prediction)*100))

# show image
im = Image.open(random_image)
im.show()



