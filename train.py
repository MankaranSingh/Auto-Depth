from utils import check_data, get_paths
from data_generator import DataGenerator2D
from loss_function import depth_loss_function
from models import get_denseDepth_model, get_simple_model
from keras.optimizers import Nadam, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from time import time

NUM_EPOCHS = 20

def main():

    if not check_data():
        exit()
    img_paths_train, img_paths_val = get_paths()
    print('Succesfully verified data...')

    train_generator = DataGenerator2D(img_paths_train['path'], './data', batch_size=1, shuffle=True, augmentation_rate=0.5)
    val_generator = DataGenerator2D(img_paths_val['path'], './data', batch_size=1, shuffle=False, augmentation_rate=0)
    print('Loaded data generators...')

    optimizer = Adam(lr=0.0001, amsgrad=True)

    model = get_simple_model()
    print('Model Loaded')
    print(model.summary())
    
    model.compile(loss=depth_loss_function, optimizer=optimizer, metrics=['mae'])
    print('Model Compiled... Starting Training...')

    tensorboard = TensorBoard(log_dir="./logs/DenseDepth/{}".format(time()), histogram_freq=1, write_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "./checkpoints/" + "DenseDepth-" + "saved-model-{epoch:03d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False)
    callbacks_list = [checkpoint, tensorboard, early_stopping]

    history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, 
                                  shuffle=True, callbacks=callbacks_list,
                                  validation_data= val_generator)
    
if __name__ == '__main__':
    main()
