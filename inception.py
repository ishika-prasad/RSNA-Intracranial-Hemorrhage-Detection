import os
import pydicom
from utils import *
import numpy as np
import pandas as pd
from preprocessing import *
from keras import backend
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split, ShuffleSplit
from keras.utils import multi_gpu_model
import keras.applications as ka
from keras.models import Sequential, Model, load_model, Input
from keras.optimizers import Adam
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout, GlobalAveragePooling2D
from keras.utils import layer_utils, Sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback
from imgaug import augmenters as iaa


# Since the dataset is huge, it is advised to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
assert 'GPU' in str(device_lib.list_local_devices())
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

#### update this section with desired values  ########
seed = 42
version = 10

# set the base directory path below
BASE_PATH = <base directory path>
TRAIN_DIR = <training images data source>
TEST_DIR = <test iamges datasource>

DENSE = 12
DROPOUT = 0.4
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
N_CLASS = 6

target_size=(224, 224, 3)

###############################################################

df = pd.read_csv(BASE_PATH + 'stage_1_train.csv').rename(columns={'Label': 'label'})
df[['id', 'img', 'subtype']] = df['ID'].str.split('_', n=3, expand=True)
df['filename'] = df['id'] + '_' + df['img'] + '.dcm'
df['path'] = BASE_PATH + TRAIN_DIR + df['filename']

bdf = pd.read_feather(DICOM_PATH).rename(columns={'SOPInstanceUID': 'filename'}).assign(filename=lambda x: x['filename']+'.dcm')[['filename', 'img_pct_window']]
df = df.merge(bdf, on='filename', how='left').drop(['ID'], axis=1)

df = df.loc[:, ["label", "subtype", "filename"]].drop_duplicates(['filename', 'subtype'])
df = df.set_index(['filename', 'subtype']).unstack(level=-1).droplevel(0, axis=1)

test_df = pd.read_csv(BASE_PATH +'/stage_1_sample_submission.csv')
test_df["Image"] = test_df["ID"].str.slice(stop=12)
test_df["Diagnosis"] = test_df["ID"].str.slice(start=13)

test_df = test_df.loc[:, ["Label", "Diagnosis", "Image"]]
test_df = test_df.set_index(['Image', 'Diagnosis']).unstack(level=-1)






class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=target_size, img_dir='stage_1_train_images/'):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = target_size
        self.img_dir= BASE_PATH + img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        if self.labels is not None: # for training phase we undersample and shuffle
            # keep probability of any=0 and any=1
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def read_dicom(self, path, target_size):
        dcm = pydicom.dcmread(path)

        try:
            img = image_preprocessing(dcm, target_size=target_size)
        except:
            img = np.zeros(target_size)

        return img
            
    def augment_img(self, image): 
        augment_img = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Fliplr(0.25),
            iaa.Flipud(0.25)])
        image_aug = augment_img.augment_image(image)
        return image_aug

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))
        
        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.augment_img(self.read_dicom(self.img_dir+ID, self.img_size))
                Y[i,] = self.labels.loc[ID].values
        
            return X, Y
        
        else:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.read_dicom(self.img_dir+ID+'.dcm', self.img_size)
            
            return X


class PredictionCheckpoint(Callback):
    
    def __init__(self, test_df, valid_df, batch_size=BATCH_SIZE, input_size=target_size):
        
        self.test_df = test_df
        self.test_images_dir = BASE_PATH + TEST_DIR
        self.batch_size = batch_size
        self.input_size = input_size
        
    def on_train_begin(self, logs={}):
        self.test_predictions = []
        
    def on_epoch_end(self,batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, img_dir=TEST_DIR), verbose=2)[:len(self.test_df)])


def build_model(input_shape, pretrained_model=None):
    net = pretrained_model(include_top=False, input_shape=input_shape)
    
    model = Sequential()
    model.add(net)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dense(DENSE, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(N_CLASS, activation='sigmoid'))
    model = multi_gpu_model(model)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[weighted_categorical_loss])
    return model

train_idx, valid_idx = train_test_split(df.index, test_size=0.01, random_state=42)
def lr_scheduler(epoch, lr):
    decay_rate = 0.8
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr


train_generator = DataGenerator(train_idx, labels=df.loc[train_idx], batch_size=BATCH_SIZE, img_size=target_size)
model_path = 'InceptionV3_{epochs}.h5'
pred_history = PredictionCheckpoint(test_df, df.loc[valid_idx])

callbacks = [
    LearningRateScheduler(lr_scheduler, verbose=1),
    ModelCheckpoint(filepath='inceptionv2-{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto'),
    pred_history
]
    
model = build_model(pretrained_model=ka.InceptionV3, input_shape=(224, 224, 3))
history = model.fit_generator(train_generator, callbacks=callbacks, epochs=10, verbose=1, use_multiprocessing=False, workers=1)

test_df.iloc[:, :] = np.average(pred_history.test_predictions, axis=0, weights=[0, 1, 2, 3, 5, 6])
test_df = test_df.stack().reset_index()
test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

test_df.to_csv('submission.csv', index=False)
