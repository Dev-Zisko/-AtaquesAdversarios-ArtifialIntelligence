import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K
from keras.preprocessing import image
from PIL import Image

import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K
from keras.preprocessing import image
from PIL import Image

iv3 = InceptionV3()

#iv3.summary()

path_img = "C://Users//Atlas//Desktop//imagenes//persona.jpg"

x = image.img_to_array(image.load_img(path_img, target_size=(299, 299)))

# Cambio de rango, 0 a 255 -> -1 a 1
x /= 255
x -= 0.5
x *= 2

x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])

y = iv3.predict(x)

decode_predictions(y)
#Hasta aqui ya predice la imagen colocada, a partir de ahora implementamos ataques adversarios

inp_layer = iv3.layers[0].input
out_layer = iv3.layers[-1].output

#Numero de la clase que queremos que salga
target_class = 951

loss = out_layer[0, target_class]

grad = K.gradients(loss, inp_layer)[0]

optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss])

adv = np.copy(x)

pert = 0.01

max_pert = x + 0.01
min_pert = x - 0.01

cost = 0.0

while cost < 0.95:
  gr, cost = optimize_gradient([adv, 0])
  adv += gr
  adv = np.clip(adv, min_pert, max_pert)
  adv = np.clip(adv, -1, 1)

hacked_img = np.copy(adv)

adv /= 2
adv += 0.5
adv *= 255

plt.imshow(adv[0].astype(np.uint8))
plt.show()

path_hack = "C://Users//Atlas//Desktop//imagenes//hacked.png"

#Con jpgs y otras resoluciones no suele engañar a la IA pero siempre va a depender de lo robusto del programa
im = Image.fromarray(adv[0].astype(np.uint8))
im.save(path_hack)