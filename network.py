import tensorflow as tf
import numpy as np
import keras
from keras import backend as K


class DescriptorCNN(tf.keras.Sequential):
    def __init__(self, arg1):
        print(str(arg1) + '!')
        super(DescriptorCNN, self).__init__()
        # self.input_shape = (1,)

        # self.W = tf.keras.backend.variable([8.0],name="W")

        self.add( tf.keras.layers.Dense(5, activation=tf.nn.relu) )


L2_dist = lambda x:K.abs(x[0]-x[1])


# fake data...
input_shape = (1,);
left_in = tf.keras.layers.Input(input_shape)
right_in = tf.keras.layers.Input(input_shape)

my_net = DescriptorCNN((1,));

left_out = my_net(left_in);
# right_out = my_net(right_in);
# both_out = keras.layers.merge([left_out, right_out], output_shape=lambda x: x[0]);
#




# silly test
fake_data_left = [1.0];
fake_data_right = [2.0];
fake_label = [0];
my_net.compile("Adam", loss="categorical_crossentropy")
out = my_net.predict(x=fake_data_left);
print(out)
