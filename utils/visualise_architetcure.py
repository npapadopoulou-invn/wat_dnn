import tensorflow as tf 
import keras 
from tensorflow.keras.utils import plot_model


# This is to visualise the architecture 

model = keras.models.load_model("/home/npapadopoulou/wat/outputs/eval_mltools/n46_9_build2_simple6rej8_noseed_scaled_lr001/model_run0/model.keras")
model.summary()
plot_model(model, to_file="/home/npapadopoulou/wat/outputs/n49_6_build2_simple6rej8_noseed_scaled_lr001/model_structure.png", show_shapes=True)


