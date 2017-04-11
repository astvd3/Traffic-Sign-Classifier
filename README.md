# Traffic-Sign-Classifier
DIY Autonomous Cars 101: Recognizing Street Signs
This is part of my article series.
The project involves deep learninng to classify street signs.
Full link to the blog post can be found here,



### Install

This project requires **Python 2.7** with the following library installed:
- [numpy](http://www.numpy.org/)
- [sklearn](http://scikit-learn.org/stable/install.html)
- [Keras](https://keras.io/)
- [Theano](https://deeplearning.net/software/theano/)

### Data

The dataset used here is from Gernman Traffic Sign Benchmark website http://benchmark.ini.rub.de/?section=home&subsection=news

### Run

`python attempt2.py` should run the program properly, but if it won't it means you haven't set your directories properly.

Your directories should be in the following fashion.

Path
|
|attempt2.py
|utils.py
|vgg16.py
|
|GTSRB  --->  |train
              |valid
              |results
              |sample (Optional)
              
Train folder will be the one containing the images with all classes, valid folder will be filled by the code.

      
