# File Organization
## 1. File Structure
### 1.1 googlenet_model contains the the googlenet model, LRN, and pool helper, where LRN and pool helper are used to build the googlenet model. Moreover, the googlenet model code is modified based on the code provided in this repository: https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
### 1.2 main usage contains config and data loader file, which are used to help make predictions to the test set, on images, and using webcam. The webcam classifictaion is developed based on this link: https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
### 1.3 Lastly, pickle files, example test images, and weights are provided in the corresponding folder
## 2. Package dependencies
### 2.1 Python 3.6
### 2.2 tensorflow-gpu 1.14 and keras 2.3.1
### 2.3 no specific version requirements on packages numpy, cv2, PIL, imageio
## 3. Repository usage
### 3.1 clone this repository
### 3.2 put the correct weights, pickles, and test images path into config file
### 3.3 run webcam prediction file to get real-time classification, performs better if have pure white background, link of youtube demonstration video: https://youtu.be/uG0b3mEjA1Q, or bilibili demonstration video: https://www.bilibili.com/video/BV1Zb4y1X7d7
### 3.4 run test set prediction file to get the test set accuracy, around 94% on 239 images
### 3.4 run image prediction file to randomly predict images under the test_image folder
