""""
Copy of wat.py in oop style
                                    
                  )                 
 (  (       )  ( /(           (     
 )\))(   ( /(  )\())   `  )   )\ )  
((_)()\  )(_))(_))/    /(/(  (()/(  
_(()((_)((_)_ | |_    ((_)_\  )(_)) 
\ V  V // _` ||  _| _ | '_ \)| || | 
 \_/\_/ \__,_| \__|(_)| .__/  \_, | 
                      |_|     |__/  
"""

import numpy as np
import os
import pandas as pd
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from utils.eval_only import ConfusionMatrix
from utils.plots import plot_snippets, plot_multitrain
import gc

# from focal_loss import SparseCategoricalFocalLoss

from tensorflow.keras import initializers


label_map = {"NONE": 0, "Double_Tap": 1, "Triple_Tap": 2}
labels = list(label_map.values())
label_list = list(label_map.keys())
num_classes = len(labels)

class WatTrainer:
    def __init__(self, name, learning_rate=0.01, batch_size=512, epochs=10000,
                 seed=None, jerk=False, both=False, scheduled=False, class_balancing=True, 
                 focal_loss=False, add_gain=True, 
                 he = True):
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.jerk = jerk
        self.both = both
        self.scheduled = scheduled
        self.class_balancing = class_balancing
        self.all_snippets = False
        self.focal_loss = focal_loss
        self.add_gain = add_gain
        self.he = he

        self.initializer= initializers.HeUniform() if self.he else initializers.GlorotUniform()

        self.model = None
        self.history = None
        self.setup_database()
        self.setup_paths()
        self.set_seed()

    def setup_database(self):
        self.data_paths = {
            "x_train": "/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/x_train.npy",
            "y_train": "/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/y_train.npy",
            "x_test": "/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/x_test.npy",
            "y_test": "/home/npapadopoulou/snippets_folder/simple6_snippets_ped150_rej8_flag/ML_accel/y_test.npy",
        }
    def setup_paths(self):
        self.paths = {
            "output": f"/home/npapadopoulou/wat/outputs/{self.name}",
            "model_eval": f"/home/npapadopoulou/wat/outputs/eval_mltools/{self.name}",
            "results": f"/home/npapadopoulou/results_wat/{self.name}"
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def set_seed(self):
        if self.seed is not None:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)
        else:
            print("No seed set: Random training.")

    def load_npy_data(self, x_path, y_path):
        print(f"Loading data from {x_path} and {y_path}...")
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y

    def load_data(self, x_path, y_path, add_gain = True, jerk=False, both=False):
        X = np.load(x_path)
        y = np.load(y_path)

        if add_gain:
            X = np.multiply(X, 256)

        if self.jerk: 
            jerk = np.diff(X, axis=1, prepend=X[:, :1, :])
            if self.both:
                X = np.concatenate((X, jerk), axis=-1)
            else: 
                X = jerk
        return X.astype('float32'), y.astype('int32')
    
    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(8, kernel_size=3, strides=1, padding='same',kernel_regularizer=regularizers.l1_l2(1e-4), kernel_initializer=self.initializer),
            BatchNormalization(), 
            Activation('relu'),

            Conv1D(8, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4), kernel_initializer=self.initializer),
            BatchNormalization(), 
            Activation('relu'),

            Dense(16, activation = 'relu', kernel_regularizer =regularizers.l1_l2(1e-4), kernel_initializer=self.initializer),
            Dropout(rate = 0.2, seed =self.seed),

            GRU(10, return_sequences=True, reset_after=True , recurrent_activation='sigmoid',
                # recurrent_initializer="orthogonal",
                recurrent_initializer='glorot_uniform', # 'he_normal' or 'glorot_uniform'
                kernel_regularizer=regularizers.l1_l2(1e-3, 1e-4),  kernel_initializer=self.initializer, seed = self.seed),
            BatchNormalization(),

            Dense(len(label_map), activation='softmax',kernel_regularizer=regularizers.l2(1e-4))
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    
    def train(self):
        start_time = datetime.now()
        ascii_border("Loading data...")
        X_train, y_train = self.load_data(self.data_paths["x_train"], self.data_paths["y_train"])
        print(f"Train data loaded: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        X_test, y_test = self.load_data(self.data_paths["x_test"], self.data_paths["y_test"])
        print(f"Test data loaded: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

        ascii_border("Preparing class weights per sample...")
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array(labels), y=y_train.flatten())
        class_weight_dict = dict(zip(labels, class_weights))
        sample_weight = np.vectorize(class_weight_dict.get)(y_train)

        ascii_border("Building model...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        self.model.save_weights(os.path.join(self.paths["output"], "initial.weights.h5"))

        log_dir = os.path.join("/home/npapadopoulou/wat/outputs/logs", datetime.now().strftime(f"{self.name}_%Y-%m-%d_%H-%M-%S"))
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Callbacks
        class BestModelCB(tf.keras.callbacks.Callback):
            best_val_acc = -1
            def on_epoch_end(self, epoch, logs=None):
                if logs["val_accuracy"] > self.best_val_acc:
                    self.best_weights = self.model.get_weights()
                    self.best_val_acc = logs["val_accuracy"]
        best_model_cb = BestModelCB()

        callbacks = [best_model_cb, tensorboard_cb]
        if self.scheduled:
            warmup_scheduler = WarmupThenExpDecayScheduler(after_epoch=self.epochs // 2, exp_decay_rate=0.996, verbose=1)
            callbacks.append(warmup_scheduler)
        
        ascii_border("Training model...")
        self.history = self.model.fit(
            X_train, y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_data=(X_test, y_test), 
            sample_weight = sample_weight if self.class_balancing else None, 
            callbacks=callbacks)
        
        self.model.set_weights(best_model_cb.best_weights)
        self.model.save(os.path.join(self.paths["output"], "model.keras"))
        ascii_border("Evaluating model...")
        self.evaluate(X_test, y_test)
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.total_seconds() / 3600
        print(f"Training completed in {hours:.2f} hours, for {self.epochs} epochs")
    

    def evaluate(self, X_test, y_test):
        y_pred_probs = self.model.predict(X_test) #  Proba per frame for each class , (N, 1600, 3)
        threshold = 0.7
        y_pred = np.argmax(y_pred_probs, axis=-1) #(N, 1600)
        y_pred_max_probs = np.max(y_pred_probs, axis=-1)
        y_pred = np.where(y_pred_max_probs >= threshold, y_pred, 0) # Where the model is NOT confident enough, I will put it as "NONE"

        np.save(f"{self.paths['model_eval']}/y_pred.npy", y_pred_probs)
        np.save(f"{self.paths['model_eval']}/y_test.npy", y_test)

        y_pred_flat = y_pred.flatten()
        y_test_flat = y_test.flatten()

        ascii_border("Classification Report")
        print(classification_report(y_test_flat, y_pred_flat, target_names=label_list, labels=labels, zero_division=0))

        ascii_border("Confusion Matrix")
        cm = ConfusionMatrix(name=self.name, label_list=label_list, labels=labels)
        cm.get_cm()
        print(cm)
        plt.savefig(f"{self.paths['output']}/confusion_matrix_th{threshold}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.paths['output']}/accuracy_plot.png")

        # get_cm(name=self.name, label_list=label_list, labels=labels)
        plot_snippets(X_test, y_test, self.name, y_pred_probs, y_pred, self.all_snippets)


if __name__ == "__main__":
    run_all = [
        {"name":  "z6_5_he_simple6_rej8_0.01_512_1000_recuinit_glorot", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_6_he_simple6_rej8_0.01_512_10000", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_7_he_simple6_rej8_0.01_512_10000", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_8_he_simple6_rej8_0.01_512_10000", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_9_he_simple6_rej8_0.01_512_10000", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_10_he_simple6_rej8_0.01_512_10000", "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
        # Larger learning rate
        {"name":  "z6_11_he_simple6_rej8_0.1_512_10000", "learning_rate":0.1, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_12_he_simple6_rej8_0.1_512_10000", "learning_rate":0.1, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_13_he_simple6_rej8_0.1_512_10000", "learning_rate":0.1, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_14_he_simple6_rej8_0.1_512_10000", "learning_rate":0.1, "batch_size":512, "epochs": 10000, "seed": None},
        {"name":  "z6_15_he_simple6_rej8_0.1_512_10000", "learning_rate":0.1, "batch_size":512, "epochs": 10000, "seed": None},
    ]
    histories_for_plotting =[]
    for config in run_all:
        tf.keras.backend.clear_session()
        gc.collect()
        
        ascii_border(f"Started training for: {config['name']}")
        trainer = WatTrainer(**config)
        trainer.train()

        histories_for_plotting.append((config["name"], trainer.history))
        trn_nb = config["name"].split("_")[0]                       #name of the folder with multi train accuracy plots

    plot_multitrain(histories_for_plotting, trn_nb)

    if __name__ == "__main__":
        run_all = [
            {"name":  "z7_1_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_2_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_3_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_4_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_5_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_6_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_7_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_8_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_9_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_10_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_11_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_12_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_13_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_14_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            {"name":  "z7_15_glorot_simple6_rej8_0.01_512_10000", 'he':False, "learning_rate":0.01, "batch_size":512, "epochs": 10000, "seed": None},
            
        ]
        histories_for_plotting =[]
        for config in run_all:
            tf.keras.backend.clear_session()
            gc.collect()
            
            ascii_border(f"Started training for: {config['name']}")
            trainer = WatTrainer(**config)
            trainer.train()

            histories_for_plotting.append((config["name"], trainer.history))
            trn_nb = config["name"].split("_")[0]                       #name of the folder with multi train accuracy plots

        plot_multitrain(histories_for_plotting, trn_nb)



