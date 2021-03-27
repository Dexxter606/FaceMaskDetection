import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout

train_data_path = "/home/poindexxter/PycharmProjects/pythonProject/train"
test_data_path = "/home/poindexxter/PycharmProjects/pythonProject/test"

'''this is the augmentation configuration we will use for training 
 it generates more images using below parameters'''
training_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
                                      height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                      fill_mode='nearest')
training_data = training_datagen.flow_from_directory(train_data_path, target_size=(200, 200), batch_size=128,
                                                     class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data = test_datagen.flow_from_directory(test_data_path, target_size=(200, 200), batch_size=128,
                                             class_mode='binary')
print(training_data.class_indices)



# Building cnn model
model=tensorflow.keras.Sequential()
model.add(tensorflow.keras.Input(shape=(200, 200, 3)))  # 250x250 RGB images
model.add(tensorflow.keras.layers.Conv2D(32, 5,  activation="relu"))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(4,4)))
model.add(tensorflow.keras.layers.Conv2D(64, 4,  activation="relu"))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(3,3)))
model.add(tensorflow.keras.layers.Conv2D(128, 3,  activation="relu"))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(units=128, activation='relu'))
model.add(tensorflow.keras.layers.Dropout(0.1))
model.add(tensorflow.keras.layers.Dense(units=256, activation='relu'))
model.add(tensorflow.keras.layers.Dropout(0.25))
model.add(tensorflow.keras.layers.Dense(units=2, activation='softmax'))
model.summary()




model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# tran_model
history = model.fit(training_data, epochs=50, verbose=1, validation_data=test_data)
model.save('model_last.h5',history)
