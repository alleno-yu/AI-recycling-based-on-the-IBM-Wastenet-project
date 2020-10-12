from transfer_learning.model_building import create_model
import cv2
import numpy as np
from PIL import Image
from collections import deque

model = create_model()

model.load_weights(r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\model\Final_Model.h5")
video = cv2.VideoCapture(0)
waste_class = ["cardboard", "glass", "metal", "paper", "plastic"]
Q = deque(maxlen=64)

while True:
        _, frame = video.read()

        output = frame.copy()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224, 224))
        img_array = np.array(im)
        img_array = img_array.transpose((2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array)[2][0]
        Q.append(prediction)
        results = np.array(Q).mean(axis=0)


        argmax = np.argmax(results)
        label = waste_class[argmax]
        probability = max(results)*100

        text = "waste type: {}".format(label)
        probability = "probability: {:.2f}".format(probability)

        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 0, 255), 5)
        cv2.putText(output, probability, (35, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 0, 255), 5)




        cv2.imshow("Output", output)

        key=cv2.waitKey(1)
        if key == ord('q'):
                break

video.release()
cv2.destroyAllWindows()