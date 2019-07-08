# Facenet_FR
## Face recognition through Facenet Model in PyTorch
FaceNet is a Deep Learning architecture consisting of convolutional layers based on GoogLeNet inspired inception models.It returns a 512 dimensioanl vectr embedding for each face

## Description

A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiples methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database

Here,we have used mtcnn to detect and align the faces from images.

We have used Haarcascade Classifier,to detect faces in a webcam,since haarcascade is faster than mtcnn.Hence,it is required that the faces be clearly visible while testing.

It is coded in Jupyter Notebook.Running the cell below "getting embeddings" is mandatory.For Face recognition through images or videos,run the respective cells.

## Result
![result](upload.png)
