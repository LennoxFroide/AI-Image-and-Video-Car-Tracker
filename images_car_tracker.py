#we import the cv2 library.
import cv2 as cv
from random import randrange

#The test image
image_file = 'car_image.jpg'

#We convert the image into a suitable version for the library.
img = cv.imread(image_file)

"""This is a pre-trained neural network from cars.xml on Github"""
car_classifier_file = 'cars.xml'

"""we create the car tracker using cascade class function and the cars.xml neural network."""
car_tracker = cv.CascadeClassifier(car_classifier_file)

"""We convert the image to grey scale. This is an optimization process. The compiler is dealing with only one chanel instead of all 3."""
car_grey_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#This will display the grey scaled image
#cv.imshow('Our cars', car_grey_scale)

"""Car detection stage.This will print rectangular coordinates from where each detected car is in the image. It gives an array output ex: [ 563  657   45   45]."""

cars = car_tracker.detectMultiScale(car_grey_scale)
#print(cars)

"""Drawing the tracking rectangles around the cars."""
for (x,y,w,h) in cars:
   cv.rectangle(img,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)), 5)

cv.imshow('Our cars', img)


key = cv.waitKey(100)


print('Code complete!!!')