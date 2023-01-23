## **Importierte** **Bibliotheken**
"""

import tensorflow as tf
from tensorflow import keras 
import numpy as np
from matplotlib import pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
import cv2 as cv
from tensorflow.keras.datasets import mnist

"""## **Dataset laden** """

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalisieren Sie die Bilder
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ändern Sie die Form der Bilder, damit sie in das CNN passen
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

"""## **Modell aufbauen**

### Sequentielles aufbauen
"""

beispiel_modell = Sequential()
beispiel_modell.add(InputLayer(input_shape=(28,28,1)))

beispiel_modell.add(Conv2D(32, (3,3), activation='relu'))
beispiel_modell.add(MaxPooling2D(pool_size=(2,2)))
beispiel_modell.add(Dropout(0.1))

beispiel_modell.add(Conv2D(32, (3,3), activation='relu'))
beispiel_modell.add(MaxPooling2D(pool_size=(2,2)))
beispiel_modell.add(Dropout(0.1))

beispiel_modell.add(Conv2D(16, (3,3), activation='relu'))
beispiel_modell.add(MaxPooling2D(pool_size=(2,2)))
beispiel_modell.add(Dropout(0.1))

beispiel_modell.add(Flatten())

beispiel_modell.add(Dense(512, activation='relu'))
beispiel_modell.add(Dense(10, activation='softmax'))

"""### Kompilieren"""

beispiel_modell.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

"""### Modell *fitten*"""

beispiel_modell.fit(x_train, y_train, batch_size=20, epochs=2, validation_split=0.2)

"""## **Test ob das Netzwerk funktioniert**"""

test_bild = x_test[11].reshape(-1, 28, 28, 1)
test_bild_darstellung = test_bild.reshape(28,28)
plt.imshow(test_bild_darstellung)

pred = beispiel_modell.predict(test_bild)
max_wert= np.argmax(pred)
print("Das Bild zeigt eine: ", max_wert)