import numpy as np
import os
import pandas as pd
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, GRU, Bidirectional, Dropout, BatchNormalization, LayerNormalization ,TimeDistributed, Dense, Activation, MaxPooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from utils.ascii import ascii_border
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import cowsay
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint # type: ignore
import random
from utils.lr_schedulers import WarmupThenExpDecayScheduler
from utils.eval_only import get_cm
from utils.plots import plot_snippets, plot_multitrain
import gc

# from focal_loss import SparseCategoricalFocalLoss

from tensorflow.keras import initializers

# tf.keras.mixed_precision.set_global_policy('mixed_float16')
##########################################
label_map = {
    "NONE": 0,
    "Double_Tap": 1,
    "Triple_Tap": 2,
}
num_classes = len(set(label_map.values()))
labels = [0, 1, 2]

def load_npy_data(x_path, y_path, scaled, jerk=False, both=False):
    X = np.load(x_path) 
    y = np.load(y_path) 

    if scaled:
        X = np.multiply(X, 256)

    if jerk: 
        jerk = np.diff(X, axis=1, prepend=X[:, :1, :])
        if both:
            X = np.concatenate((X, jerk), axis=-1)
        else: 
            X = jerk
    return X.astype('float32'), y.astype('int32')

# # ----- complicated ---- #
# def build_model(input_shape, num_classes, learning_rate, focal_loss):
#     model = Sequential([
#         Conv1D(16, kernel_size=8, strides=1, padding='same', input_shape=input_shape, kernel_regularizer =regularizers.l2(1e-4)),
#         BatchNormalization(),
#         Activation('relu'),

#         Conv1D(16, kernel_size=5, strides=1, padding='same', kernel_regularizer =regularizers.l2(1e-4)),
#         BatchNormalization(),
#         Activation('relu'),

#         Conv1D(8, kernel_size=5, strides=1, padding='same', kernel_regularizer =regularizers.l2(1e-4)),
#         BatchNormalization(),
#         Activation('relu'),

#         Dropout(0.2),
        
#         GRU(8, return_sequences=True),
#         BatchNormalization(),

#         # TimeDistributed(Dense(num_classes, kernel_regularizer =regularizers.l2(1e-4))),
#         Dense(num_classes, kernel_regularizer =regularizers.l2(1e-4)),
#         Activation('softmax')
#     ])
    
#     model.compile(optimizer=Adam(learning_rate = learning_rate),  loss=SparseCategoricalFocalLoss(gamma= [0, 0, 2]) if focal_loss else 'sparse_categorical_crossentropy', metrics=['accuracy'])  #loss='sparse_categorical_crossentropy'
#     return model

####### simple 4 ########## t2 
def build_model2_orig(input_shape, num_classes, learning_rate, focal_loss = False):
    model = Sequential([
        Conv1D(16, kernel_size=5, strides=1, padding='same',kernel_regularizer=regularizers.l2(1e-4), input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer =regularizers.l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),

        Dropout(0.3),
        
        Bidirectional(GRU(12, return_sequences=True)),
        LayerNormalization(),

        Dense(num_classes, kernel_regularizer =regularizers.l2(1e-5), activation='softmax'),
    ])
    
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model


# Build 2 
def build_model(input_shape, num_classes, learning_rate, focal_loss = False):
    initializer =  tf.keras.initializers.HeUniform()
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        BatchNormalization(),
        Activation('relu'),

        Dense(16, activation = 'relu', kernel_regularizer =regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        Dropout(0.2),
        
        GRU(10, return_sequences=True, kernel_regularizer=regularizers.l1_l2(1e-5),  kernel_initializer=initializer, recurrent_initializer="orthogonal"), 

        # TimeDistributed(Dense(num_classes, activation = 'softmax')),
        Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(1e-4)),
    ])
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

def build_model2(input_shape, num_classes, learning_rate, focal_loss = False):
    initializer =  tf.keras.initializers.HeUniform()
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        # BatchNormalization(),
        Activation('relu'),

        Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        # BatchNormalization(),
        Activation('relu'),

        Dense(16, activation = 'relu', kernel_regularizer =regularizers.l1_l2(1e-4), kernel_initializer=initializer),
        Dropout(0.2),
        
        GRU(10, return_sequences=True, kernel_regularizer=regularizers.l1_l2(1e-5),  kernel_initializer=initializer, recurrent_initializer="orthogonal"), 

        # TimeDistributed(Dense(num_classes, activation = 'softmax')),
        Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(1e-4)),
    ])
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

###### One conv1 ##########
# def build_model(input_shape, num_classes, learning_rate, focal_loss):
#     model = Sequential([
#         Input(shape=input_shape),
#         Conv1D(8, kernel_size=9, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4)),
#         BatchNormalization(),
#         Activation('relu'),

#         Dense(16, activation = 'relu', kernel_regularizer =regularizers.l1_l2(1e-4)),
#         Dropout(0.2),
        
#         GRU(8, return_sequences=True, kernel_regularizer=regularizers.l1_l2(1e-5)),

#         # TimeDistributed(Dense(num_classes, kernel_regularizer =regularizers.l2(1e-4))),
#         Dense(num_classes, activation = 'softmax'),
#     ])
    
#     model.compile(optimizer=Adam(learning_rate = learning_rate), loss=SparseCategoricalFocalLoss(gamma=2) if focal_loss else 'sparse_categorical_crossentropy', metrics=['accuracy'])  #loss='sparse_categorical_crossentropy'
#     return model

def evaluate_and_visualize(model, X_test, y_test, label_list, history, name):
    
    os.makedirs(f"/home/npapadopoulou/wat/{name}", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/eval_mltools/wat/{name}", exist_ok=True)
    
    ascii_border("Predicting on test set...", "double")

    y_pred_probs = model.predict(X_test)   #  probability distributions for each class , (N, 1600, 4)
    
    threshold = 0.7  # Only keeps predictions where the model is 70% confident
    y_pred = np.argmax(y_pred_probs, axis=-1)  #(N, 1600)
    y_pred_max_probs = np.max(y_pred_probs, axis=-1)    
    y_pred = np.where(y_pred_max_probs >= threshold, y_pred, 0) # Where the model is NOT confident enough, I will put it as "NONE"

    np.save(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_pred.npy", y_pred_probs)
    np.save(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_test.npy", y_test)

    y_pred_flat = y_pred.flatten()  #
    y_test_flat = y_test.flatten()

    ascii_border("Classification Report", "double")
    print(classification_report(y_test_flat, y_pred_flat, target_names=label_list, labels=labels, zero_division=0))

    ascii_border("Confusion Matrix", "double")
    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels, normalize= 'true')
    print(cm)
    get_cm(name=name, label_list=label_list, labels=labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"/home/npapadopoulou/wat/{name}/confusion_matrix_th{threshold}.png")
   
    # Snippets ensuring all classes are included - saved in results_wat
    plot_snippets(X_test, y_test, name, y_pred_probs, y_pred, all=False)
    
    ascii_border("Visualizing Accuracy vs Epochs graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label = 'Training Accuracy', color = "blue")
    plt.plot(history.history['val_accuracy'], label = 'Test Accuracy', color = "red")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/npapadopoulou/wat/{name}/accuracy_plot.png")
    
def train_model(name, fine_tuning = False,focal_loss = False,scaled = True, learning_rate = 0.001, phasetraining = False, jerk=False, both=False, epochs=1000, batch_size=258, class_balancing = True, seed = None, scheduled = False):

    if seed is not None:
        # os.environ['PYTHONHASHSEED']=str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    os.makedirs(f"/home/npapadopoulou/wat/{name}", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/wat/eval_mltools/{name}", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/wat/eval_mltools/{name}/model_run0/", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/results_wat/{name}", exist_ok=True)
    

    start_time = datetime.now()
    x_test_path ="/home/npapadopoulou/snippets_folder/simple7_snippets_ped150_rej7_flag/ML_accel/x_test.npy"
    y_test_path = "/home/npapadopoulou/snippets_folder/simple7_snippets_ped150_rej7_flag/ML_accel/y_test.npy"
    x_train_path="/home/npapadopoulou/snippets_folder/simple7_snippets_ped150_rej7_flag/ML_accel/x_train.npy"
    y_train_path="/home/npapadopoulou/snippets_folder/simple7_snippets_ped150_rej7_flag/ML_accel/y_train.npy"

    # x_val_path="/home/npapadopoulou/snippets_folder/simple4_snippets_ped150_rej7/ML_accel/x_test.npy"
    # y_val_path="/home/npapadopoulou/snippets_folder/simple4_snippets_ped150_rej7/ML_accel/y_test.npy"

    label_list =["NONE","Double_Tap","Triple_Tap"]

    print("Loading train data...")
    X_train, y_train = load_npy_data(x_train_path, y_train_path, scaled, jerk, both)
    print(f"\nTrain data loaded: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")

    y_train_flat = y_train.flatten()
    class_weights =  compute_class_weight(class_weight="balanced", classes= np.array(labels), y=y_train_flat)
    class_weight_dict = dict(zip(labels, class_weights))
    ascii_border(f"Class weights:\n{class_weight_dict}")
    sample_weight = np.vectorize(class_weight_dict.get)(y_train)    # Take an array like y_train and return it with the sample weigths for the train

    # print("Loading validation data...")
    # X_val, y_val = load_npy_data(x_val_path, y_val_path, jerk, both)
    # print(f"\n Validation data loaded: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")

    print("Loading test data...")
    X_test, y_test = load_npy_data(x_test_path, y_test_path, scaled, jerk, both)
    print(f"\n Test data loaded: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    
    ascii_border("Building model...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes, learning_rate = learning_rate, focal_loss=focal_loss)
    model.save_weights(f"/home/npapadopoulou/wat/{name}/init.weights.h5") 
    model.summary()

    ########## Callbacks ###########
    class BestModelCB(tf.keras.callbacks.Callback):
        best_val_acc=-1
        def on_epoch_end(self, epoch, logs=None):
            val_acc=logs["val_accuracy"]
            if val_acc>self.best_val_acc:
                self.best_weights = self.model.get_weights()
                self.best_val_acc=val_acc
    best_model_cb=BestModelCB()
    # Tensorflow callaback
    log_dir = os.path.join("/home/npapadopoulou/wat/logs", datetime.now().strftime(f"{name}_%Y-%m-%d_%H-%M-%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    decay_scheduler_cb = WarmupThenExpDecayScheduler(after_epoch = epochs // 2, exp_decay_rate=0.996, verbose=1) 

    callbacks = [best_model_cb, tensorboard_cb] + ([decay_scheduler_cb] if scheduled else [])
    ascii_border("Training...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), sample_weight = sample_weight if class_balancing else None, callbacks=callbacks)
    model.set_weights(best_model_cb.best_weights) # get best weights

    print("Saving the model...")
    model.save(f'/home/npapadopoulou/wat/{name}/model.keras')
    model.save(f"/home/npapadopoulou/wat/eval_mltools/{name}/model_run0/model.keras")
    if fine_tuning:
        new_model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes, learning_rate = 0.005)
        new_model.load_weights(f'/home/npapadopoulou/wat/{name}/model.keras')
        new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), sample_weight = sample_weight if class_balancing else None, callbacks=callbacks, shuffle=False if seed is not None else True)

    ascii_border("Evaluating...")
    model.evaluate(X_test, y_test)
    evaluate_and_visualize(model, X_test, y_test, label_list, history, name)

    end_time = datetime.now()
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600
    cowsay.cow(f"All done, model finished training!\nThis took {hours:.2f} hours for {epochs} epochs")

    return history

def train_model2(name, fine_tuning = False,focal_loss = False,scaled = True, learning_rate = 0.001, phasetraining = False, jerk=False, both=False, epochs=1000, batch_size=258, class_balancing = True, seed = None, scheduled = False):

    if seed is not None:
        # os.environ['PYTHONHASHSEED']=str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    os.makedirs(f"/home/npapadopoulou/wat/{name}", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/wat/eval_mltools/{name}", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/wat/eval_mltools/{name}/model_run0/", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/results_wat/{name}", exist_ok=True)
    

    start_time = datetime.now()
    x_test_path ="/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/x_test.npy"
    y_test_path = "/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/y_test.npy"
    x_train_path="/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/x_train.npy"
    y_train_path="/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/y_train.npy"

    # x_val_path="/home/npapadopoulou/snippets_folder/simple4_snippets_ped150_rej7/ML_accel/x_test.npy"
    # y_val_path="/home/npapadopoulou/snippets_folder/simple4_snippets_ped150_rej7/ML_accel/y_test.npy"

    label_list =["NONE","Double_Tap","Triple_Tap"]

    print("Loading train data...")
    X_train, y_train = load_npy_data(x_train_path, y_train_path, scaled, jerk, both)
    print(f"\nTrain data loaded: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")

    y_train_flat = y_train.flatten()
    class_weights =  compute_class_weight(class_weight="balanced", classes= np.array(labels), y=y_train_flat)
    class_weight_dict = dict(zip(labels, class_weights))
    ascii_border(f"Class weights:\n{class_weight_dict}")
    sample_weight = np.vectorize(class_weight_dict.get)(y_train)    # Take an array like y_train and return it with the sample weigths for the train

    # print("Loading validation data...")
    # X_val, y_val = load_npy_data(x_val_path, y_val_path, jerk, both)
    # print(f"\n Validation data loaded: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")

    print("Loading test data...")
    X_test, y_test = load_npy_data(x_test_path, y_test_path, scaled, jerk, both)
    print(f"\n Test data loaded: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    
    ascii_border("Building model...")
    model = build_model2(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes, learning_rate = learning_rate, focal_loss=focal_loss)
    model.save_weights(f"/home/npapadopoulou/wat/{name}/init.weights.h5") 
    model.summary()

    ########## Callbacks ###########
    class BestModelCB(tf.keras.callbacks.Callback):
        best_val_acc=-1
        def on_epoch_end(self, epoch, logs=None):
            val_acc=logs["val_accuracy"]
            if val_acc>self.best_val_acc:
                self.best_weights = self.model.get_weights()
                self.best_val_acc=val_acc
    best_model_cb=BestModelCB()
    # Tensorflow callaback
    log_dir = os.path.join("/home/npapadopoulou/wat/logs", datetime.now().strftime(f"{name}_%Y-%m-%d_%H-%M-%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    decay_scheduler_cb = WarmupThenExpDecayScheduler(after_epoch = epochs // 2, exp_decay_rate=0.996, verbose=1) 

    callbacks = [best_model_cb, tensorboard_cb] + ([decay_scheduler_cb] if scheduled else [])
    ascii_border("Training...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), sample_weight = sample_weight if class_balancing else None, callbacks=callbacks)
    model.set_weights(best_model_cb.best_weights) # get best weights

    print("Saving the model...")
    model.save(f'/home/npapadopoulou/wat/{name}/model.keras')
    model.save(f"/home/npapadopoulou/wat/eval_mltools/{name}/model_run0/model.keras")
    if fine_tuning:
        new_model = build_model2(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes, learning_rate = 0.005)
        new_model.load_weights(f'/home/npapadopoulou/wat/{name}/model.keras')
        new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), sample_weight = sample_weight if class_balancing else None, callbacks=callbacks, shuffle=False if seed is not None else True)

    ascii_border("Evaluating...")
    model.evaluate(X_test, y_test)
    evaluate_and_visualize(model, X_test, y_test, label_list, history, name)

    end_time = datetime.now()
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600
    cowsay.cow(f"All done, model finished training!\nThis took {hours:.2f} hours for {epochs} epochs")

    return history

###############################################################################################################################################################

# bs 512 lr 0.01
if __name__ == "__main__":
    run_all = [

    {"name": "n77_withbatch_9_simple7rej7_noseed_scaled_lr001_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_10_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_1_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_2_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_3_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_4_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_5_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n77_withbatch_6_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 1000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
]

    histories_for_plotting = []
    for config in run_all:
        tf.keras.backend.clear_session()
        gc.collect()
        ascii_border(f"Started training for: {config['name']}")

        history = train_model(**config)
        histories_for_plotting.append((config["name"], history))
        trn_nb = config["name"].split("_")[0]                       #name of the folder with multi train accuracy plots

    plot_multitrain(histories_for_plotting, trn_nb)

if __name__ == "__main__":
    run_all = [
    {"name": "n76_nobatch_7_simple7rej7_noseed_scaled_lr001_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_8_simple7rej7_noseed_scaled_lr001_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_9_simple7rej7_noseed_scaled_lr001_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_10_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_1_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_2_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_3_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_4_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_5_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
    {"name": "n76_nobatch_6_simpl76rej7_noseed_scaled_lr002_1000", "epochs": 10000, "batch_size": 512, "seed": None, "learning_rate": 0.01},
]

    histories_for_plotting = []
    for config in run_all:
        tf.keras.backend.clear_session()
        gc.collect()
        ascii_border(f"Started training for: {config['name']}")

        history = train_model2(**config)
        histories_for_plotting.append((config["name"], history))
        trn_nb = config["name"].split("_")[0]                       #name of the folder with multi train accuracy plots

    plot_multitrain(histories_for_plotting, trn_nb)

