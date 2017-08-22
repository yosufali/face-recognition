# Face Detection


###### A piece of coursework for the Computer Vision module I took at university.

## Aim

The goal of this project was, using supervised machine learning, to create 4 different classifiers that had the ability to classify a new unseen image. The classes were labeled from `1` to `37`, one for each student on the module.

I chose to implement the following classifiers:

- Support Vector Machine using Bag of Words features
- Support Vector Machine using History of Oriented Gradients features
- Feedforward Neural Network using Bag of Words features
- Feedforward Neural Network using History of Oriented Gradients features

The classifiers were trained using images of students taking the module.

## Running
The `RecogniseFace.m` file is the starting point of the program. The function called is `P(I, featureType, classifierName)`
, where I is the path of the new image to be classified, featureType is the user's chosen feature type (HOG or BAG) and classifier name is the user's chosen classification method (SVM or FNN).

This function returns `P`, a Nx3 matrix, where N is the number of faces found in the provided image `I`, and the columns are `id`: the predicted identity (class) of the found face, `x`, and `y`, the central co-ordinates of the found face within the image.

The best performing classifier of those attempted was the SVM with BAG features with 98% accuracy.

---

This project only works with images of students who took the module, since those are the faces that it was was trained to classify.
