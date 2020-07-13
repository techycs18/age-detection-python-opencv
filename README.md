# Real-Time age Detection using OpenCV
### Description :
The project is made for the purpose of knowledge. It simple detection the faces and performs age detection on the face. The ages are classified into 8 groups<br/>

`Age groups: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53) & (60-100)` <br/>


#### Prerequisites :
  ###### Required 
  - Python Programming Language 
  - Machine Learning algorithms
  - Convolutional Neural Network 

  ###### Not compulsory(given below), but pior knowledge would be beneficial 
  - Caffe-Based Deep Learning models 
  - OpenCV for Python

#### Installations :
> - [Install Python](https://www.python.org/downloads/)<br/>
> - [Install OpenCV for windows](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html) <br/>
> - [Install OpenCV for Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) <br/>
> - pip install imutils <br/>
> - pip install numpy <br/>

#### Procedure
- For age detection on image run:
`python image_age_detection.py --image images/1.png --face face_detector --age age_detector`
- For real-time(webc-cam) age detection run:
 `python real_time_age_detection.py --face face_detector --age age_detector`
