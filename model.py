import os
import csv
import cv2
import numpy as np
import sklearn
import math, random
from random import shuffle
from PIL import Image
import matplotlib.pyplot as plt

samples = []
#with open('../../opt/carnd_p3/data/driving_log.csv') as csvfile:
with open('data_training_track/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    print('csv file read successfully')
    samples.pop(0)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def flip_image(img):
    return (np.fliplr(img))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    #print('sample size = ', num_samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        steer_correction = 0.2

            
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steer_correction
                steering_right = steering_center - steer_correction
            

                #current_path = '../../opt/carnd_p3/data/IMG/'
                current_path = 'data_training_track/IMG/'
                img_center = np.asarray(Image.open(current_path + batch_sample[0].split('/')[-1]))
                flip_yn = random.randint(0,1)
                if flip_yn == 1:
                    img_center = flip_image(img_center)
                    steering_center = -steering_center
                img_left = np.asarray(Image.open(current_path + batch_sample[1].split('/')[-1]))
                flip_yn = random.randint(0,1)
                if flip_yn == 1:
                    img_left = flip_image(img_left)
                    steering_left = -steering_left
                img_right = np.asarray(Image.open(current_path + batch_sample[2].split('/')[-1]))
                flip_yn = random.randint(0,1)
                if flip_yn == 1:
                    img_right = flip_image(img_right)
                    steering_right = -steering_right
            
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
            X_train = np.array(images)
            y_train = np.array(angles)
            # Augmenting random images
            for x in range( int(batch_size/4)):
                img_number = random.randint(1,batch_size)
                #X_train[img_number],y_train[img_number] = flip_image(X_train[img_number],y_train[img_number])
            
            
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Cropping2D
#from keras.layers.pooling

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255-0.5))
model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#for layer in model.layers:
#    print(layer.output_shape)

model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size)*3, validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size)*3, epochs=3, verbose=1)

model.save('model_04.h5')

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

