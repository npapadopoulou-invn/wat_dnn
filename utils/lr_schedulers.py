import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, LearningRateScheduler # type: ignore

class WarmupThenExpDecayScheduler(Callback):
    def __init__(self, after_epoch=5000, exp_decay_rate=0.996, verbose=1):
        super().__init__()
        self.after_epoch = after_epoch
        self.exp_decay_rate = exp_decay_rate
        self.verbose = verbose
        self.initial_lr = None
        self.scale = 500
        

    def on_train_begin(self, logs=None):
        # Safely get the initial learning rate
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            self.initial_lr = float(tf.keras.backend.get_value(lr))
        else:
            self.initial_lr = float(lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.after_epoch:
            return
        decayed_lr = self.initial_lr * (self.exp_decay_rate ** ((epoch - self.after_epoch)/self.scale))
        # Set the LR robustly, whether it's a float, variable, or something else
        lr = self.model.optimizer.learning_rate
        try:
            tf.keras.backend.set_value(lr, decayed_lr)
        except Exception:
            # Fallback: direct assignment if not a tf.Variable
            self.model.optimizer.learning_rate = decayed_lr

        if self.verbose:
            print(f"\nOn epoch {epoch}: Learning rate set to {decayed_lr:.6f}")
