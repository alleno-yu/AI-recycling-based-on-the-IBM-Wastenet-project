import cv2
import numpy as np
from PIL import Image
from collections import deque

from googlenet_model.googlenet import create_googlenet
from main_usage.config import waste_class, weights_path

# build googlenet model
model = create_googlenet(weights_path)


# use the code provided in this link: https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
# average out video classification results

# capture video using the webcam and cv library
video = cv2.VideoCapture(0)
# average window size
Q = deque(maxlen=64)

while True:
        _, frame = video.read()

        output = frame.copy()

        # Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224, 224))
        img_array = np.array(im)
        img_array = img_array.transpose((2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        # Calling the predict method on model to predict on the image
        prediction = model.predict(img_array)[2][0]
        Q.append(prediction)
        results = np.array(Q).mean(axis=0)

        # get predicted label and probability
        argmax = np.argmax(results)
        label = waste_class[argmax]
        probability = max(results)*100

        # display them
        text = "waste type: {}".format(label)
        probability = "probability: {:.2f}".format(probability)

        # put text on the cv2 displayed video
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 0, 255), 5)
        cv2.putText(output, probability, (35, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 0, 255), 5)


        cv2.imshow("Output", output)

        # quit key is set to q
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

# destroy window
video.release()
cv2.destroyAllWindows()