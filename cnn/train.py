### ---------- Import Relevand Libraries ----------

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from model import create_model
from math import ceil

model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Prepare the data

# using flow_from_directory(), generate batches of image data (and their labels)
batch_size = 40

train_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            zoom_range=0.1, # randomly zooming inside pictures
            horizontal_flip=False,
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift

val_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            zoom_range=0.1, # randomly zooming inside pictures
            horizontal_flip=False,
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift


# generate batches of augmented image data, given the path of the original images

# instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory)
# this generator will read pictures found in subfolders and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../train_split/',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
val_generator = val_datagen.flow_from_directory(
        '../validation_split/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


# use the generators above to train the model
# stop when the val accuracy stops getting better

filepath="_weights.h5"

checkpoint_callback = ModelCheckpoint(
                        filepath,
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stop_callback = EarlyStopping(
                monitor='val_acc',
                patience=3,
                mode='max') 

callbacks_list = [checkpoint_callback, early_stop_callback]

history = model.fit_generator(
            train_generator,
            steps_per_epoch=(ceil(len(train_generator.filenames) // batch_size)),
            epochs=50,
            validation_data=val_generator,
            validation_steps=(ceil(len(val_generator.filenames) // batch_size)),
            callbacks=callbacks_list)



