import time

import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.constraints import max_norm
from keras.layers import Activation, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Input, Conv1D, Flatten, MaxPooling1D, Reshape, UpSampling1D
from keras.layers import LSTM, Dense, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.models import Sequential
from tensorflow import keras
from tensorflow.python.keras.models import Sequential


# =============================================================================
# MODEL TOPOLOGIES
# =============================================================================

# =============================================================================
# THE CNN MODEL TOPOLOGY
# =============================================================================


def CNNModel(train_data_x, train_data_y, nConvLayers, kernel_size, stride, kernel_constraint, nUnits, initialFilters,
             dropout1, dropout2):
    shape = (train_data_x.shape[1:])
    input1 = Input(shape=shape)
    output_size = train_data_y.shape[1]
    # convolutions
    x = input1
    MAXFILTERNUM = 128
    for i in range(nConvLayers):

        nFilters = MAXFILTERNUM if initialFilters * (2 ** (i)) > MAXFILTERNUM else initialFilters * (2 ** (i))
        # inside = 3 if i>1 else 2
        inside = 2
        # TODO zaki promenio k = 11 if i == 0 else kernel_size
        k = kernel_size
        for temp in range(inside):
            x = Conv1D(filters=nFilters,
                       kernel_size=k,
                       padding='same',
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_constraint=max_norm(kernel_constraint),
                       name='Conv1x{}x{}_{}_{}'.format(k, nFilters, i, temp))(x)

            x = Activation('relu', name='ReLu_{}_{}'.format(i, temp))(x)
            x = BatchNormalization(name='Batch_{}_{}'.format(i, temp))(x)

            x = MaxPooling1D(pool_size=2, padding='same', name='MaxPooling1x2_{}_{}'.format(i, temp))(x)

    x = Dropout(dropout1)(x)

    # Fully connected
    x = Flatten()(x)

    x = Dense(nUnits,
              kernel_constraint=max_norm(1),
              kernel_initializer='he_normal')(x)
    x = Activation('relu', name='reLU_dense')(x)
    x = Dropout(dropout2)(x)

    x = Dense(output_size)(x)
    x = Activation('softmax', name='Softmax')(x)

    model = Model(input1, x)
    return model


# =============================================================================
# SEQUENTIAL MODEL TOPOLOGY
# =============================================================================
def CNNSequentialModel(train_data_x, train_data_y, nConvLayers, kernel_size, stride, kernel_constraint, nUnits,
                       initialFilters,
                       dropout1, dropout2):
    shape = (train_data_x.shape[1:])
    output_size = train_data_y.shape[1]
    model = Sequential()
    model.add(Conv1D(initialFilters, kernel_size, padding='same', activation='relu', input_shape=shape))
    model.add(Conv1D(initialFilters, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(2 * initialFilters, kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(2 * initialFilters, kernel_size, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout1))
    model.add(Dense(output_size, activation='softmax'))

    return model


# =============================================================================
# LSTM MODEL TOPOLOGY
# =============================================================================
def LSTMModel(train_data_x, train_data_y, nConvLayers, kernel_size, stride, kernel_constraint, nUnits, initialFilters,
              dropout1, dropout2):
    shape = (train_data_x.shape[1:])
    output_size = train_data_y.shape[1]
    # TODO unaprediti model
    model = Sequential()

    # Recurrent layer
    model.add(LSTM(units=kernel_size, input_shape=shape, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    model.add(Dense(nUnits, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(dropout1))

    # Output layer
    model.add(Dense(output_size, activation='softmax'))

    return model


# =============================================================================
# RANDOM MODEL TOPOLOGY
# =============================================================================
def CNNRandomModel(train_data_x, train_data_y, layers, filter_size, stride, kernel_constraint, nUnits, initialFilters,
                   dropout1, dropout2):
    shape = (train_data_x.shape[1:])
    input1 = Input(shape=shape)
    output_size = train_data_y.shape[1]
    # convolutions
    x = input1
    filter_layer_num = 0
    max_filter_layer_num = 128
    conv_percent = 0.25
    activation_percent = 0.25
    normalization_percent = 0.25
    pooling_percent = 0.25
    name = 'CNNRandomModel'

    for i in range(layers):

        num = np.random.random()
        if (num < conv_percent) or (i == 0):
            nFilters = max_filter_layer_num if initialFilters * (
                    2 ** filter_layer_num) > max_filter_layer_num else initialFilters * (2 ** filter_layer_num)
            x = Conv1D(filters=nFilters,
                       kernel_size=filter_size,
                       padding='same',
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_constraint=max_norm(kernel_constraint),
                       name='Conv1x{}x{}_{}'.format(filter_size, filter_layer_num, i))(x)
            filter_layer_num = filter_layer_num + 1
            name = name + 'C'
            continue
        if num < conv_percent + activation_percent:
            x = Activation('relu', name='ReLu_{}'.format(i))(x)
            name = name + 'A'
            continue

        if num < conv_percent + activation_percent + normalization_percent:
            x = BatchNormalization(name='Batch_{}'.format(i))(x)
            name = name + 'B'
            continue
        if num < conv_percent + activation_percent + normalization_percent + pooling_percent:
            x = MaxPooling1D(pool_size=2, padding='same', name='MaxPooling1x2_{}'.format(i))(x)
            name = name + 'M'
        continue

    # EXIT layer
    x = Dropout(dropout1)(x)

    # Fully connected
    x = Flatten()(x)

    x = Dense(nUnits, kernel_constraint=max_norm(1), kernel_initializer='he_normal')(x)
    x = Activation('relu', name='reLU_dense')(x)
    x = Dropout(dropout2)(x)

    x = Dense(output_size)(x)
    x = Activation('softmax', name='Softmax')(x)

    model = Model(input1, x, name=name)
    return model


# =============================================================================
# MULTI-HEADED MODEL TOPOLOGY
# =============================================================================

def MultiHeadedModel(train_data_x, train_data_y, nConvLayers=3, kernel_size=3, stride=1, kernel_constraint=0, nUnits=64,
                     initialFilters=32, dropout1=0.6, dropout2=0.7):
    n_timesteps, n_features, n_outputs = train_data_x.shape[1], train_data_x.shape[2], train_data_y.shape[1]
    inputs = []
    flats = []
    for i in range(nConvLayers):
        # head i
        inputsi = Input(shape=(n_timesteps, n_features))
        convi = Conv1D(filters=initialFilters, kernel_size=(kernel_size + i * 2), activation='relu')(inputsi)
        dropi = Dropout(dropout1)(convi)
        pooli = MaxPooling1D(pool_size=2)(dropi)
        flati = Flatten()(pooli)

        inputs.append(inputsi)
        flats.append(flati)

    # merge
    merged = concatenate(flats)
    # interpretation
    dense1 = Dense(nUnits, activation='relu')(merged)
    dense2 = Dropout(dropout2)(dense1)
    outputs = Dense(n_outputs, activation='softmax')(dense2)
    model = Model(inputs=inputs, outputs=outputs)

    return model


# =============================================================================
# MULTI-HEADED MODEL TOPOLOGY 1
# =============================================================================

def MultiHeadedModel1(train_data_x, train_data_y, nConvLayers=3, kernel_size=3, stride=1, kernel_constraint=3,
                      nUnits=64,
                      initialFilters=32, dropout1=0.6, dropout2=0.7):
    n_timesteps, n_features, n_outputs = train_data_x.shape[1], train_data_x.shape[2], train_data_y.shape[1]
    inputs = []
    flats = []
    for i in range(nConvLayers):
        inside = 2
        k = kernel_size + i * 2
        inputs_i = Input(shape=(n_timesteps, n_features))
        x = inputs_i
        for temp in range(inside):
            x = Conv1D(filters=initialFilters,
                       kernel_size=k,
                       padding='same',
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_constraint=max_norm(kernel_constraint),
                       name='Conv1x{}x{}_{}_{}'.format(k, initialFilters, i, temp))(x)

            x = Activation('relu', name='ReLu_{}_{}'.format(i, temp))(x)
            x = BatchNormalization(name='Batch_{}_{}'.format(i, temp))(x)

            x = MaxPooling1D(pool_size=2, padding='same', name='MaxPooling1x2_{}_{}'.format(i, temp))(x)

        flati = Flatten()(x)
        inputs.append(inputs_i)
        flats.append(flati)

    # merge
    merged = concatenate(flats)
    x = Dropout(dropout1)(merged)

    # Fully connected
    x = Flatten()(x)

    x = Dense(nUnits, kernel_constraint=max_norm(1), kernel_initializer='he_normal')(x)
    x = Activation('relu', name='reLU_dense')(x)
    x = Dropout(dropout2)(x)

    x = Dense(n_outputs)(x)
    x = Activation('softmax', name='Softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


# =============================================================================
# THE CNN AUTOENCODER MODEL TOPOLOGY
# =============================================================================

def CNNModelAutoencoder(train_data_x, train_data_y, nConvLayers, kernel_size, stride, kernel_constraint, initFilters,
                        nUnits, dropout1, dropout2):  # TODO not tested

    shape = (train_data_x.shape[1:])
    input1 = Input(shape=shape)
    output_size = train_data_y.shape[1]

    # convolutions
    x = input1
    MAXFILTERNUM = 128

    for i in range(nConvLayers):

        nFilters = MAXFILTERNUM if initFilters * 2 ** i > MAXFILTERNUM else initFilters * 2 ** i
        # inside = 3 if i>1 else 2
        inside = 2
        for temp in range(inside):
            x = Conv1D(filters=nFilters,
                       kernel_size=kernel_size,
                       padding='same',
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_constraint=max_norm(kernel_constraint),
                       name='Conv1x{}x{}_{}_{}'.format(kernel_size, nFilters, i, temp))(x)

            x = Activation('relu', name='ReLu_{}_{}'.format(i, temp))(x)
            x = BatchNormalization(name='Batch_{}_{}'.format(i, temp))(x)

            x = MaxPooling1D(pool_size=2, padding='same', name='MaxPooling1x2_{}_{}'.format(i, temp))(x)

    x = Dropout(dropout1)(x)

    # Fully connected
    x = Flatten(name='flatWhite')(x)

    #    x = Dense(nUnits,
    #              kernel_constraint = max_norm(kernel_constraint),
    #              kernel_initializer = 'he_normal')(x)
    #    x = Activation('relu', name = 'reLU_dense')(x)
    #    x = Dropout(dropout2,name='FCstuffEncoded')(x)

    filtTemp = 128 if nConvLayers > 2 else (nConvLayers) * initFilters
    rShape = (int(np.ceil(train_data_x / (2 ** (2 * nConvLayers)))), filtTemp)
    #    print(rShape)
    #    x = Dense(int(np.ceil(inputShape[0]/(2**(2*nConvLayers))))*filtTemp,
    #              activation='relu',
    #              kernel_initializer='he_normal',
    #              name='denseDude')(x)
    x = Reshape(rShape)(x)
    # decoder

    for i in range(nConvLayers):

        nFilters = 128 if i < nConvLayers - 2 else initFilters * (nConvLayers - i)

        for temp in range(2):
            x = Conv1D(filters=nFilters,
                       kernel_size=kernel_size,
                       padding='same',
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_constraint=max_norm(kernel_constraint))(x)

            x = Activation('relu')(x)
            x = BatchNormalization()(x)

            x = UpSampling1D(size=2)(x)

    x = Conv1D(kernel_size=1, strides=1, filters=6,
               padding='same',
               kernel_initializer='he_normal',
               kernel_constraint=max_norm(kernel_constraint))(x)
    x = Activation('sigmoid')(x)

    model = Model(input1, x)
    features = model.get_layer('flatWhite').output
    encoder = Model(input1, features)

    return model, encoder


# utilize initialization
def CNNModelFromAutoencoder(train_data_x, train_data_y, nUnits, dropoutRate):  # TODO not tested
    encoder_file_name = './results/encoderCEOshuffleRIGHT.h5'
    encoder = load_model(encoder_file_name)
    output_size = train_data_y.shape[1]

    x = encoder.output
    x = Dense(nUnits, kernel_constraint=max_norm(1),
              kernel_initializer='he_normal',
              activation='relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(output_size, activation='softmax')(x)

    model = Model(encoder.input, x)
    return model


# =============================================================================
# FITERS
# =============================================================================

def fit_model(model, model_name, def_callbacks, train_data_x, train_data_y, validation_data_x, validation_data_y,
              epochs, batch_size):
    tic = time.time()

    # make file name
    tm = time.gmtime()
    weight_file = './results/{}.{}.{}.{}.{}.{}.h5'.format(model_name, tm[0], tm[1], tm[2], tm[3] + 1,
                                                          tm[4])

    # define callbacks
    callbacks = def_callbacks(weight_file)

    # fit model
    x = train_data_x
    y = train_data_y
    history = model.fit(x=x, y=y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(validation_data_x, validation_data_y),
                        shuffle=True)
    toc = time.time()
    print("Finished training in {} min ({} h)".format(round((toc - tic) / 60, 2), round((toc - tic) / 3600, 2)))

    # Save the weights
    # model.save_weights(str(modelName)+'.h5') # ???????

    return history


def fit_model_no_validation(model, model_name, def_callbacks, train_data_x, train_data_y, validation_data_x,
                            validation_data_y, epochs, batch_size):
    tic = time.time()

    # make file name
    tm = time.gmtime()
    weight_file = './results/{}.{}.{}.{}.{}.{}.h5'.format(model_name, tm[0], tm[1], tm[2], tm[3] + 1, tm[4])

    # define callbacks
    callbacks = def_callbacks(weight_file)

    # fit model
    x = train_data_x
    y = train_data_y
    # evaluate model
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    toc = time.time()
    print("Finished training in {} min ({} h)".format(round((toc - tic) / 60, 2), round((toc - tic) / 3600, 2)))

    # Save the weights
    # model.save_weights(str(modelName)+'.h5') # ???????

    return history


def fit_modelA(model, model_name, train_data_x, train_data_y, validation_data_x,
               validation_data_y, epochs, batch_size):  # TODO not tested
    tic = time.time()

    # make file name
    tm = time.gmtime()
    weight_file = './results/{}.{}.{}.{}.{}.{}.h5'.format(model_name, tm[0], tm[1], tm[2], tm[3] + 1, tm[4])

    # define callbacks
    callbacks = def_callbacks2(weight_file)

    train_x = train_data_x
    validation_x = validation_data_x
    # FIT THE MODEL
    history = model.autoencoder.fit(train_x, train_x,  # nije greska oba puta se uzima isto
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    callbacks=callbacks,
                                    validation_data=(validation_x, validation_x),  # XvalA,XvalA
                                    shuffle=True)

    model.autoencoder.load_weights(weight_file)
    for l1, l2 in zip(model.encoder.layers, model.autoencoder.layers[:27]):
        l1.set_weights(l2.get_weights())

    # Save the weights
    encoder_file_name = './results/encoderCEOshuffleRIGHT.h5'
    model.encoder.save(encoder_file_name)

    toc = time.time()
    print("Finished training in {} min ({} h)".format(round((toc - tic) / 60, 2), round((toc - tic) / 3600, 2)))

    return history


# =============================================================================
# CALLBACKS
# =============================================================================


def def_callbacks1(weight_file):
    checkpoint = ModelCheckpoint(weight_file,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 mode='max')
    early = EarlyStopping(monitor='val_loss',
                          patience=20,
                          verbose=1,
                          mode='min')
    #     def step_decay(epoch):
    #         initial_lrate = 0.001
    #         rate_drop = 0.25
    #         nEpochs = 5
    #         lrate = initial_lrate * math.pow(rate_drop, math.floor(epoch/nEpochs))
    #         return lrate

    #     lrate = LearningRateScheduler(step_decay, verbose = 1)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=8, min_lr=0.0000000001,
                                   verbose=1)

    return [checkpoint, early, lr_plateau]


def def_callbacks2(weight_file):
    checkpoint = ModelCheckpoint(weight_file,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 mode='min')
    early = EarlyStopping(monitor='val_loss',
                          patience=20,
                          verbose=1,
                          mode='min')
    #     def step_decay(epoch):
    #         initial_lrate = 0.001
    #         rate_drop = 0.25
    #         nEpochs = 5
    #         lrate = initial_lrate * math.pow(rate_drop, math.floor(epoch/nEpochs))
    #         return lrate

    #     lrate = LearningRateScheduler(step_decay, verbose = 1)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=8, min_lr=0.0000000001,
                                   verbose=1)

    return [checkpoint, early, lr_plateau]


def def_callbacks3(weight_file):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=weight_file,
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=20,
                                      verbose=1,
                                      mode='min')
    ]

    return callbacks


# =============================================================================
# OPTIMIZERS
# =============================================================================

def get_optimizer_adam():
    return optimizers.Adam(lr=0.001)


def get_optimizer_sgd():
    return optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)


# =============================================================================
# COMPILERS
# =============================================================================

def compile_model(model, optimizer):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return
