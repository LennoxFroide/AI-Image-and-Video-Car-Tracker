# AI-Image-and-Video-Car-Tracker
This is an application that tracks the location of cars in image and video files. It uses a pre-trained classifier provided on github by cars.xml.
A coloured image is first converted to grey scale so as to make the algorithm processes the images faster. The grey scale image is then passed through a cascade of Haar features and assessed whether it has any cars on it.
When a video file is used, each frame is isolated and it undergoes the same process that an image would go through and the classfier determines whether or not it has any cars in it.
