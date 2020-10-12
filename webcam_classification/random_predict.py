from transfer_learning.model_building import create_model
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
import os
import random

model = create_model()

model.load_weights(r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\model\Final_Model.h5")
waste_class = ["cardboard", "glass", "metal", "paper", "plastic"]
# path = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\test_set_images"
path = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\test_set_images"

images = os.listdir(path)

random_image = random.choice(images)
random_image = os.path.join(path, random_image)

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



