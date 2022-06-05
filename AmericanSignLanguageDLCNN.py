import tensorflow as tf
from sklearn import metrics
import cv2
from cv2 import *
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import keras

def Load_Images(Folder_Path,img_name):
    img = cv2.imread(os.path.join(Folder_Path,img_name))
    img2 = cv2.resize(img, (64, 64))
    imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)       
    return imgCol


    
def Load_TrainingData(alphbet,Folder_Path):
    training_data=[]
    for i in alphbet:
        Class_Number=alphbet.index(i)
        Data_Folder = os.path.join(Folder_Path, i)
        print(Data_Folder)
        for img_name in os.listdir(Data_Folder):
            try:
                img=Load_Images(Data_Folder,img_name)
                training_data.append(img)
            except Exception as e:
                pass
    return training_data

def Load_testData(alphbet,Folder_Path):
    test_data=[]
    #images=Load_Images(Folder_Path)
    i=0 
    # Class_Number=alphbet.index(i)
    for img_name in os.listdir(Folder_Path):
        if alphbet[i]=='del':
           break
        try:
           img=Load_Images(Folder_Path,img_name)
           test_data.append([img, i])
        except Exception as e:
            pass
        i+=1
    # for j in images:  
    #     if alphbet[i]=='del':
    #         break
    #     test_data.append([j, i])
    #     i+=1    
    return test_data


def vectorization (Data):
    return np.array(Data).reshape(-1,64,64,3)

def Getxy_Train (training_data):
    X_Train=[]
    Y_Train=[]
    for features,label in training_data:
        X_Train.append(features)
        Y_Train.append(label)
        X_Train=vectorization(X_Train)
        X_Train=X_Train.astype('float32')/255.0 #to normalize images
        Y_Train=keras.utils.to_categorical(Y_Train)
        Y_Train=np.array(Y_Train)
    return  X_Train,Y_Train

def Getxy_Test(testdata):
    X_Test=[]
    Y_Test=[]
    for features,label in testdata:
        X_Test.append(features)
        Y_Test.append(label)
    return X_Test,Y_Test

alphbet=['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S',
              'space','T', 'U', 'V', 'W', 'X', 'Y', 'Z']
Folder_Path="E://Level Four//First Term//Machine 2//Assignment2//DATA//asl_alphabet_train"
training_data=Load_TrainingData(alphbet,Folder_Path)
train_images, train_labels=Getxy_Train (training_data)

Folder_Path2="E://Level Four//First Term//Machine 2//Assignment2//DATA//asl_alphabet_test"
testdata=Load_testData(alphbet,Folder_Path2)
test_images, test_labels=Getxy_Test(testdata)

train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])

    # which is why you need the extra index
    plt.xlabel(alphbet[train_labels[i][1]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)



        