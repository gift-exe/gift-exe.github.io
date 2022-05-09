---
layout: post
title:  Face Mask Detection with CNNs
date:   2022-05-09 09:20:00 +0100
image:  /assets/images/blog/facemask_cnn.jpg
author: Gift
tags:   print('CNNs')
---

Hello 👋🏿
Today I'm going to show you how I made a software that can detect if a person is wearing facemask or not using Convolutional Neural Networks (CNNs)

CNNs is a class of Artificial Neural Networks that is most commonly applied to analize visual imagery. CNNs are used commonly in image and video recognition,
recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain–computer interfaces, and financial time series.

CNNs were inspired by biological processe, in that the connectivity pattern between neurons in a CNN resembles the organization of the animal visual coretx.

There a lot more to CNNs you can read more on https://en.wikipedia.org/wiki/Convolutional_neural_network.

Now we're going on to CNNs and face mask detection.

IMPLEMENTATION

we'll start by importing the modules we need for this project

    #import libraries
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
    from keras.callbacks import ModelCheckpoint


then we initialize our cnn model
using tensorflow keras Sequential which is basically just a linear stack of layers, so if the model is training,
he layers are going to be implemented step by step in the order in which they are written down (sequentially)

the first layer is the Conv2D layer (a 2D convolutional layer)
the activation function used is the "rectified linear unit" 

The MaxPooling2d just show that the convolutional layer has a stride of 2
pooling is done to fix the problem of overfitting in CNNs by "down sampling" 

    model =Sequential([Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
                       MaxPooling2D(2,2),
        
                       Conv2D(100, (3,3), activation='relu'),
                       MaxPooling2D(2,2),
        
                       Flatten(),
                       Dropout(0.5),
                       Dense(50, activation='relu'),
                       Dense(2, activation='softmax')])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


now that we're done with the model we have to collect our data to train the midel with and a test dataset for validating

    TRAINING_DIR = "./train"
    #Augmentation configuration
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                        batch_size=10, 
                                                        target_size=(150, 150))

    VALIDATION_DIR = "./test"
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                                batch_size=10, 

ImageDataGenerator lets you augment the images in real time while the model is training hence more training Image data.

checkpoint and model_history

    #creating the check point
    checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

    #model's history
    history = model.fit_generator(train_generator,
                                epochs=10,
                                validation_data=validation_generator,
                                callbacks=[checkpoint])


After we've trained our model, we can now move on to the next stage of testing the model in real time...

MODEL TESTING....

To test our model we're going to be using the opencv library. so as usual we import the libraries we need

    import cv2
    import numpy as np
    from keras.models import load_model

then we load our model and go on from there

    #load the model
    model=load_model("\model2-010.model")

    labels_dict = {0:'without mask', 1:'mask'}

    color_dict = {0:(0,0,255), 1:(0,255,0)}
    
    size = 4
    webcam = cv2.VideoCapture(0) #Use camera 0

    # We load the xml file for capturing faces
    classifier = cv2.CascadeClassifier(r'\haarcascade_frontalface_default.xml')


    while True:
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(150,150))

            #convert into array for the model
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            
            label=np.argmax(result,axis=1)[0]
        
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        # Show the image
        cv2.imshow('LIVE', im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break
    # Stop video
    webcam.release()

    # Close all started windows
    cv2.destroyAllWindows()