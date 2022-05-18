---
layout: post
title:  Face Mask Detection with CNNs
date:   2022-05-18 13:00:00 +0100
image:  /assets/images/blog/heart_disease.jpg
author: Gift
tags:   sklearn, tensorflow, LinearClassifier
---

Heyy, 
Today I'll show you a basic heart disease detection model (a model that can detect if a person has heart disease or not), using a Linear Classifier model.

Basically, a LinearClassifier is a model that uses stats to determine the group or class an object belongs to. the model makes a prediction based on the linear combination of the object characteristics.
[graph representation](/assets/images/blog/LC.jpg)  

There a lot more to CNNs you can read more on [wikipedia](https://en.wikipedia.org/wiki/Linear_classifier)  

Lets get to the implementation now:  

lets import the libraries:

    #import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    from six.moves import urllib
    import tensorflow.compat.v2.feature_column as fc
    import tensorflow as tf

alright we are good to go, lets load the data and preprocess:

    #read data and preprocess
    df_train = pd.read_csv('/content/train1.csv')
    df_eval = pd.read_csv('/content/eval1.csv')
    y_train = df_train.pop('HeartDisease')
    y_eval = df_eval.pop('HeartDisease')

    #divide different columns with respect to their data_type
    CATEGORICAL_COLUMNS = ['Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope']
    NUMERIC_COLUMNS = ['Age', 'Oldpeak']
    
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = df_train[feature_name].unique() #print all the unique values in each columns
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype= tf.float64))


"tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)"  

basically this function creates a categorical column with a list of possible values.
while a feature column is a list that is used to specify how Tensors received from the input should be transformed and comined before entering the model  

ok. Let's move on  

next we would create an input function that helps with training the model in batches and epoches  

    def make_input_function(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
        def input(): # this function will be returned in the function
            data_set = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                data_set = data_set.shuffle(1000) #randomize the order of data
            data_set = data_set.batch(batch_size).repeat(num_epochs)   #split the data into sets(batches) of 32 and repeat the process 10 times(num of epochs)
            return data_set  #returns a batch of the data_set
        return input

    train_input_function = make_input_function(df_train, y_train)  #change the format of the dataset into one that we can feed the model
    eval_input_function = make_input_function(df_eval, y_eval, num_epochs= 1, shuffle= False)  #this is for testing the model

So with this function we can successfully feed the model with the correct format of data and labels.  

The next step is to initialize the model, train and test the model for accuracy 

    #initialization of the model with tensorflow estimator
    linear_estimator = tf.estimator.LinearClassifier(feature_columns= feature_columns)

    linear_estimator.train(train_input_function)    #trainning
    result = linear_estimator.evaluate(eval_input_function)  #testing

    clear_output()    #clear console outputs
    print(result['accuracy'])       #this will show the accuracy of the model's prediction on the data_set for evaluatio
    #this will vary since the testing data_set is on shuffle


you can get the source code at my github page: [click here to visit](https://github.com/Gift-py/ML-Logistic-Regression)

If you have any issue feel free to contact me (use the contact me box below 👀). I'll definetly try to help you ✨✨  
And if you have any project idea or you wanna colaborate with me don't hesitate I'll always want to join your team 😌


Allright... That's all for now.
Byee 👋🏿







