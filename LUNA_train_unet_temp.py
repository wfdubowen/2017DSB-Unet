from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

working_path = "./LUNA/tutorial/"
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
img_rows = 512
img_cols = 512
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_1)
    drop4 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5_1 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5_2 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5_1)
    drop5 = Dropout(0.5)(conv5_2)
    # us5 = UpSampling2D(size=(2, 2))(conv5_2)
    us5 = Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))

    up6 = concatenate([drop4, us5], axis=3)
    conv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6_1)
    # us6 = UpSampling2D(size=(2, 2))(conv6_2)
    us6 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6_2))

    up7 = concatenate([conv3_2, us6], axis=3)
    conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7_1)
    # us7 = UpSampling2D(size=(2, 2))(conv7_2)
    us7 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7_2))

    up8 = concatenate([conv2_2, us7], axis=3)
    conv8_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8_1)
    # us8 = UpSampling2D(size=(2, 2))(conv8_2)
    us8 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8_2))

    up9 = concatenate([conv1_2, us8], axis=3)
    conv9_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9_1)
    conv9_3 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9_2)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9_3)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def train_and_predict(use_existing):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load(working_path + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainMasks.npy").astype(np.float32)
    imgs_test = np.load(working_path + "testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + "testMasks.npy").astype(np.float32)
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    # model = get_unet()
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights?
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')
    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reseasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=80, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
    # loading best weights from training session
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('./unet.hdf5')
    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i + 1]], verbose=1)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ", mean)
    score = model.evaluate(imgs_mask_test,imgs_mask_test_true, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    train_and_predict(False)
