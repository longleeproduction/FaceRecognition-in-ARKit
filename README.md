# FaceRecognition in ARKit

This is a simple showcase project, that detects faces using the Vision-API and runs the extracted face through a CoreML-model to identiy the specific persons.

![image of scene with face recognition](demo.gif)


## Requirements

* Xcode 9
* iPhone 6s or newer
* Machine-Learning model

## Machine-Learning model


To create your own machine-learning model, you can read our blog post ["How we created our Face-Recognition model"](https://www.novatec-gmbh.de/en/blog/created-face-recognition-model/)

The short version is:

* We trained a model in the AWS using Nvidia DIGITS
* Took a couple of hundred pictures of each person, and extracted the faces
* Also added an "unknown" category with differnent faces.
* Used a pretrained model fine-tuned for face-recognition.



## Acknowledgements

* [3D-Text](https://github.com/hanleyweng/CoreML-in-ARKit)


How we created our Face-Recognition model

As described in our previous posts, we created an ARKit-App with Face-Recognition.
I will explain how we created our Face-Recognition model.

Where to start?
Apple’s machine learning framework CoreML supports Keras and Caffe for neural network machine learning.
Never heard of these before and done anything with machine learning, I started with a Keras tutorial:



A simple dogs vs. cats example.

The results and the performance were not that good. Especially with a Macbook and ATI-Graphics.

Nvidia DIGITS and AWS
There are a couple of tools, that are beginner-friendly.
One of is Nvidia DIGITS, which offers

Integrated frameworks Caffe, Torch and TensorFlow
Pre-trained models such as AlexNet and GoogLeNet
Easy to use UI
Rather than installing it on my local machine with no GPU support, I went for a AWS instance.
The newest image I found in the AWS was DIGITS 4. Even if DIGITS 6 is available.

Setup
1. Create an instance

Create a new instance from the AWS Marketplace

2. Select Instance Type

For our usecase the g2.2xlarge is good enough.

3. Configure Security Group

For security reasons we change the Source for all ports to My IP.

4. The instance is running

Keep the instance only running if you are using it, otherwise it will get expensive.

5. Open the Public DNS URL

After a couple of minutes you should see the DIGITS UI

Now we need data
For our training data, we did a little photo-session and shot a couple of hundred photos for each person:



Extracting
To get our face-detection running, we need to extract the same face/head proportion for our training photos as we do later in the app.
We created a litte script to extract the faces, downscale and save the new files:
https://gist.github.com/m-ruhl/fee63420105c9f0a4d7e0657a2ac7156

The unkown
Aside from our specific persons, we need an unknown category. So that other faces are not falsely identified as one of our classifications.
For this we searched the web, and downloaded lots of different faces.

Start the training
1. Upload our images

Now we need to upload our images to create a dataset.
Connect with SFTP to our instance. Use the username ubuntu and your AWS pem-file.

Upload the classifications in the folder data

2. Create Dataset

Go back to the DIGITS UI and create a new classification dataset.
You may need to enter a username. Choose as you like.


Set the image size to 227 x 227
Caffe’s default cropped size is 227 x 227. Otherwise greater images will be cropped.
Resize Transformation to Crop
In order to keep the aspect ratio, we crop the images.
Training images to /home/ubuntu/data
Image Encoding to JPEG(lossy)
To save space.

Now we have our dataset to create our ML model.

3. Train our model

Navigate to the main page and create a new classification model.

We reduce the training epochs to 15 for a faster result and the learning rate to 0.001 
Choose the AlexNet as network


The accuracy is not great, but it is a start.

4. Use a pre-trained model

The standard AlexNet model is not optimised for face-recognition.
There a many pre-trained model available, some of the are listed here:
https://github.com/BVLC/caffe/wiki/Model-Zoo#berkeley-trained-models
In the end we chose to use FaceDetection-CNN model.

Upload the pretrained model to /home/ubuntu/models/
Go back to the DIGITS UI and click on the Clone Job of our previous run.

Click on Customize and


Define the pre-trained model /home/ubuntu/models/face_alexnet.caffemodel


With our new run, we get way better results


A quick check validates our model. It correctly recognises philipp

Integrate the model in iOS
1. Download model


Download the model from the lastest epoch

2. Install CoreMLTools

To use our Caffe ML-model in our iOS-App we have convert it to a CoreML compatible format.
Apple provides a tool for this: coremltools
You can install it with:  pip install -U coremltools

3. Create convert script

We need to write a little python script to convert our model. It’s based on examples and documentation available here: https://apple.github.io/coremltools/
convert.pyPython
import coremltools

# Convert a caffe model to a classifier in Core ML
coreml_model = coremltools.converters.caffe.convert(('snapshot_iter_360.caffemodel',
													 'deploy.prototxt',
													 'mean.binaryproto'),
													  image_input_names = 'data',
													  class_labels = 'labels.txt',
													  is_bgr=True, image_scale=227.)

# Now save the model
coreml_model.save('faces_model.mlmodel')
1
2
3
4
5
6
7
8
9
10
11
12
import coremltools
 
# Convert a caffe model to a classifier in Core ML
coreml_model = coremltools.converters.caffe.convert(('snapshot_iter_360.caffemodel',
													 'deploy.prototxt',
													 'mean.binaryproto'),
													  image_input_names = 'data',
													  class_labels = 'labels.txt',
													  is_bgr=True, image_scale=227.)
 
# Now save the model
coreml_model.save('faces_model.mlmodel')

You may need to adjust the file-name of the caffemodel
Save this to the same folder as the model and run it with:  python convert.py

4. Integration to project

We only need to copy faces_model.mlmodel to our Showcase-Project

All compile errors in Xcode should disappear.

Final result
