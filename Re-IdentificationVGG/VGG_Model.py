from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from tensorflow.python.keras.models import save_model, load_model

from scipy.io import loadmat
import numpy as np
import copy
import math


#BUILD MODEL ARCHITECTURE
# building block of the VGG model :
def convblock(cdim, nb, bits=3):
    L = []

    for k in range(1, bits + 1):
        convname = 'conv' + str(nb) + '_' + str(k)
        L.append(Convolution2D(filters=cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))

    L.append(MaxPooling2D((2, 2), strides=(2, 2)))

    return L


def vgg_face_blank():
    # Model initialization
    mdl = Sequential()
    # First layer is a dummy-permutation = Identity to specify input shape
    mdl.add(Permute((1, 2, 3), input_shape=(224, 224, 3)))  # WARNING : 0 is the sample dim

    # Model body
    for l in convblock(64, 1, bits=2):
        mdl.add(l)

    for l in convblock(128, 2, bits=2):
        mdl.add(l)

    for l in convblock(256, 3, bits=3):
        mdl.add(l)

    for l in convblock(512, 4, bits=3):
        mdl.add(l)

    for l in convblock(512, 5, bits=3):
        mdl.add(l)

    # Model head
    mdl.add(Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6'))
    mdl.add(Dropout(0.5))

    mdl.add(Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7'))
    mdl.add(Dropout(0.5))

    mdl.add(Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8'))
    mdl.add(Flatten())
    mdl.add(Activation('softmax'))

    return mdl

#Feeding the model weights


def copy_mat_to_keras(kmodel):

    kerasnames = [lr.name for lr in kmodel.layers]
    data = loadmat('vgg_face_matconvnet/data/vgg_face.mat', matlab_compatible=False, struct_as_record=False)
    net = data['net'][0, 0]
    l = net.layers
    description = net.classes[0, 0].description

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    #prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            #print matname
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            #f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            #f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])


def features(featmodel, crpimg, transform=False):
    # transform=True seems more robust but I think the RGB channels are not in right order
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:, :, 0] -= 129.1863
        imarr[:, :, 1] -= 104.7624
        imarr[:, :, 2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        # imarr[:, :, 0] = aux[:, :, 2]
        # imarr[:, :, 2] = aux[:, :, 0]

        # imarr[:,:,0] -= 129.1863
        # imarr[:,:,1] -= 104.7624
        # imarr[:,:,2] -= 93.5940

    # imarr = imarr.transpose((2,0,1))
    imarr = np.expand_dims(imarr, axis=0)

    fvec = featmodel.predict(imarr)[0, :]
    fvec =  fvec[0, 0, :]
    # normalize
    normfvec = math.sqrt(fvec.dot(fvec))
    return fvec / normfvec


# layer -7 to output fc6
def vgg_model():
    facemodel = vgg_face_blank()
    copy_mat_to_keras(facemodel)
    featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-7].output)
    save_model(featuremodel, 'vgg_model_FC6.h5', overwrite=True,include_optimizer=False)








