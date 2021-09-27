#Train cross validation
from sklearn.model_selection import KFold
from model.model_builder import *
from data_preparation import config
from data_preparation.data_processing import build_dataset

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, EarlyStopping

C = config.Config()

image_size = C.im_size
depth_size = C.depth_size

model = build_model(image_size, depth_size)

kf = KFold(n_splits=5, random_state=19, shuffle=True)

def training(scans, labels, name, EPOCHS, BATCH_SIZE, save_checkpoint):
    for fold, (train_index, val_index) in enumerate(kf.split(scans)):
        print(f'Fold {fold} of {name}'.format(fold))
        train_dataset, val_dataset = build_dataset(scans, labels, \
                                                      train_index, val_index,\
                                                      BATCH_SIZE, BATCH_SIZE,\
                                                      EPOCHS)
        compile_model(model)
    #     epoch_end = epoch_end()
        early_stop = EarlyStopping(monitor='val_loss', patience=6)
        model_checkpoint = ModelCheckpoint(f'{save_checkpoint}/model_fold_{fold}_{name}.h5',
                                      monitor = 'val_loss',
                                      verbose=1,
                                      save_best_only=True)
        model.fit(train_dataset, 
                steps_per_epoch=int(len(train_dataset)/BATCH_SIZE),
                epochs = EPOCHS,
                callbacks=[model_checkpoint, early_stop],
                validation_data=val_dataset,
                validation_steps=int(len(val_dataset)/BATCH_SIZE))
