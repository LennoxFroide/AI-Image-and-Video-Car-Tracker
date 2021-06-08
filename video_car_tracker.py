import cv2 as cv
from random import randrange

#The test video clip
video_file = 'Car_dashcam_feed.mp4'

#The function videocapture allows us to gain access to our webcam video feed
video = cv.VideoCapture(video_file)

while True:
    """The function webcam.read() converts our file to a suitable format for the cv2 library. It returns a tuple. The tuple contains a boolean which is stored in the variable successful_frame_read and a frame from the video which is stored in variable frame."""
    successful_frame_read, frame = video.read()
    
    #This is a check measure. It allows us to identify whether the video has been read successfully.
    if not successful_frame_read:
        break

    #A pre-trained neural network from cars.xml on Github
    cars_classifier_file = 'cars.xml'
    

    #We create the car tracker here
    video_car_tracker = cv.CascadeClassifier(cars_classifier_file)
    

    #We convert the video to grey scale to allow the algorithm to work faster
    video_cars_grey_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #This displays our video
    cv.imshow('Car Tracking system', video_cars_grey_scale)


    """The detectMultiScale functions returns a multi-dimensional array with the coordinates of the position of he vehicles in each frame of the video."""
    video_cars = video_car_tracker.detectMultiScale(video_cars_grey_scale)
    #print(video_cars)


    #Drawing tracking rectangles around the identified vehicles in each frame
    for (x,y,w,h) in video_cars:
        cv.rectangle(frame,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),5)

        #This displays our video
        cv.imshow('Car Tracking system', frame)

    

    key = cv.waitKey(1)


#This allows us to quit the program instanly by pressing x on the keyboard.

    if key==88 or key==120:
        break



"""This informs the operating system that the program has finished using memory and that the resoures should be directed to other programs"""
video.release()




print('Code complete!!!!')