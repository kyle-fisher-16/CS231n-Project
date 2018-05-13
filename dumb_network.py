from keras.layers import Input, Conv2D, Lambda, merge, Dense, dot, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.initializers import Constant
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os

#def W_init(shape,name=None):
#  """Initialize weights as in paper"""
#  values = rng.normal(loc=0,scale=1e-2,size=shape)
#  return K.variable(values,name=name)
##//TODO: figure out how to initialize layer biases in keras.
#def b_init(shape,name=None):
#  """Initialize bias as in paper"""
#  values=rng.normal(loc=0.5,scale=1e-2,size=shape)
#  return K.variable(values,name=name)

img_W = 7;
img_H = 7;
input_shape = (2,img_W,img_H)
siamese_input = Input(input_shape)
batch_sz = 10;

left_input = Lambda(lambda x: x[:, 0, :, :],
                    output_shape=(batch_sz,img_W,img_H),
                    )(siamese_input);

right_input = Lambda(lambda x: x[:, 1, :, :],
                     output_shape=(batch_sz,img_W,img_H),
                     )(siamese_input);

# make the network
convnet = Sequential()
convnet.add(Dense(3, activation='relu', input_shape=input_shape, kernel_initializer=Constant(value=1), bias_initializer=Constant(value=0)))

# encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
# merge two encoded inputs with the l1 distance between them
abs_diff_lambda = lambda x: K.abs(x[0]-x[1])
abs_diff = merge([encoded_l, encoded_r], mode = abs_diff_lambda, output_shape=lambda x: x[0])
dist_sq = dot([abs_diff, abs_diff], axes=1, normalize=False)

# loss function
def l2_loss(y_true, y_pred):
  # y_true should be boolean (int)... 0 if non-corresponding, 1 if matched.
  # y_pred is distance squared from siamese network
  
  # todo: piece-wise loss from the paper
  return K.sqrt(y_pred);

# instantiate the network
siamese_net = Model(output=dist_sq, inputs=siamese_input)

optimizer = Adam(0.006) # default was 0.00006
siamese_net.compile(loss=l2_loss,optimizer=optimizer)

N = 100;

fake_data = 1.0 * np.ones((N, 2, img_W, img_H))
fake_data[:,0,:, :] = 10.0;
fake_labels = np.zeros((N,),dtype="uint8")

#network_out = siamese_net.predict(x=fake_data)
#print network_out
#loss_output = siamese_net.evaluate(x=[fake_data], y=fake_labels)
#print loss_output
siamese_net.fit(x=fake_data, y=fake_labels, batch_size=batch_sz, epochs=30);

