import cv2
import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
f = os.listdir("image-dataset")
j = len(f)
print j
#resizes the image to a constant size
def change():
    i = 1
    while i<=j:
        i = str(i)
        path = "image-dataset/"+i+".png"
        print path
        i = int(i)
        image = cv2.imread(path)
        print image
        image = cv2.resize(image, (24,24))
        #print image.shape
        cv2.imwrite(path, image)
        i=i+1
#makes the 3d image to 2d
def grayscale():
    i=1
    while i<=j:
        i = str(i)
        path = "image-dataset/"+i+".png"
        print path
        i = int(i)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, gray)
        i=i+1
#goes to each an every image make them 1d array then write the image data to csv file with incrementing filenames such as 1.csv
def check():
    i=1
    while i<=j:
        i = str(i)
        a = i+".csv"
        path = "image-dataset/"+i+".png"
        i = int(i)
        image = cv2.imread(path)
        #print path
        array = np.array(image)
        b = array.ravel()
        print b
        print"f"
        np.set_printoptions(suppress=True)
        np.savetxt(a, [b], delimiter=',', fmt='%d')
        i=i+1
#demo function
def check1():
    path = "image-dataset/31.png"
    image = cv2.imread(path)
    #print image
    array = np.array(image)
    print array
    print array.shape
    b = array.ravel()
    print b.shape
    print b
    np.set_printoptions(suppress=True)
    #np.savetxt('dataset.csv', [b], header="", delimiter=',', fmt='%d')
#writing all the files to a single file
def process():
    i=1
    while i<=31:
        i = str(i)
        a = i+".csv"
        i = int(i)
        file = open(a, "r")
        data = file.read()
        file.close()
        fi = open("dataset.csv", "a+")
        fi.write(data)
        fi.close()
        i=i+1
#check1()
#all the cool stuff
dataset = pd.read_csv("dataset.csv")
print dataset
np.set_printoptions(suppress=True)
dataset = dataset.to_numpy()
print dataset
results = pd.read_csv("results.csv",usecols=[0])
np.set_printoptions(suppress=True)
results = results.to_numpy()
print results
relu = 'relu'
model = Sequential()
model.add(Dense(200,input_dim=1728,activation='relu'))
model.add(Dense(180,activation=relu))
model.add(Dense(160,activation=relu))
model.add(Dense(140,activation=relu))
model.add(Dense(120,activation=relu))
model.add(Dense(100,activation=relu))
model.add(Dense(80,activation=relu))
model.add(Dense(60,activation=relu))
model.add(Dense(40,activation=relu))
model.add(Dense(20,activation=relu))
model.add(Dense(10,activation=relu))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dataset, results, epochs=50, batch_size=10)
accuracy = model.evaluate(dataset,results)
print accuracy
model.save("model1.h5")
test = pd.read_csv("test.csv")
np.set_printoptions(suppress=True)
test = test.to_numpy()
res = model.predict_classes(test)
print res
model = load_model('model.h5')
model.summary()
score = model.evaluate(dataset, results, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
