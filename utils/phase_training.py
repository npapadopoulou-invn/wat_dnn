# else:
#         # Model checkpoint callback --- Could be use later for refining, useless now 
#         checkpoint_filepath = f'/home/npapadopoulou/wat/{name}/checkpoint.model.keras'
#         model_checkpoint_callback = ModelCheckpoint(
#             filepath=checkpoint_filepath,
#             monitor='val_accuracy',
#             mode='max',
#             save_best_only=True, 
#             save_weights_only=False)
        
#         # Phase 1
#         callbacks1 = [model_checkpoint_callback, tensorboard_cb]
#         ascii_border("Training...")
#         history1 = model.fit(X_train, y_train, epochs= epochs, batch_size=batch_size, 
#                             validation_data=(X_test, y_test),
#                             sample_weight = sample_weight if class_balancing else None, 
#                             callbacks=callbacks1)
#         # Phase 2
#         print(" Loading best model from Phase 1...")
#         model = tf.keras.models.load_model(checkpoint_filepath)

#         decay_scheduler_cb2 = WarmupThenExpDecayScheduler(after_epoch = 0, exp_decay_rate=0.996, verbose=1)
#         callbacks2 = [model_checkpoint_callback, tensorboard_cb, decay_scheduler_cb2]
#         history = model.fit(X_train, y_train, epochs=epochs//2, batch_size=batch_size, 
#                             validation_data=(X_test, y_test),
#                             sample_weight = sample_weight if class_balancing else None, 
#                             callbacks=callbacks2)
        
#         print("Saving the final model...")
#         model.save(f'/home/npapadopoulou/wat/{name}/model.keras')
#         model.save(f"/home/npapadopoulou/wat/eval_mltools/{name}/model.keras")







from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense, Dropout, GRU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes, learning_rate, focal_loss=False):
    
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(8, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l1_l2(1e-4),
               kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(8, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l1_l2(1e-4),
               kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),

        Dense(16, activation='relu',
              kernel_regularizer=regularizers.l1_l2(1e-4),
              kernel_initializer='he_normal'),
        Dropout(0.2),

        GRU(10, return_sequences=True,
            kernel_regularizer=regularizers.l1_l2(1e-5),
            kernel_initializer='he_normal',  # applies to GRU kernel
            recurrent_initializer='he_normal'),  # for recurrent connections

        Dense(num_classes, activation='softmax',
              kernel_initializer='he_normal'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalFocalLoss(gamma=[0, 0, 1]) if focal_loss else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
