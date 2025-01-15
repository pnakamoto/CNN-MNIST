from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist  # Para carregar o dataset MNIST
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Redimensionar os dados para incluir o canal (28x28x1) e normalizar
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# Codificação one-hot para as classes
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construção do modelo sequencial
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Configuração do otimizador e compilação do modelo
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# Callback para reduzir o learning rate se não houver melhorias
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

# Parâmetros de treinamento
batch_size = 32
epochs = 10

# Treinamento do modelo
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
    callbacks=[learning_rate_reduction]
)

# Extração das métricas do treinamento
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
range_epochs = range(1, len(acc) + 1)

# Visualização da acurácia
plt.style.use('default')
plt.plot(range_epochs, val_acc, label='Acurácia no conjunto de validação')
plt.plot(range_epochs, acc, label='Acurácia no conjunto de treino', color="r")
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(loc="lower right")
plt.show()





from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo salvo
model = load_model('meu_modelo_mnist.h5')

# Substitua pelo caminho correto da sua imagem
img_path = '3.png'

# Carregar e pré-processar a imagem
image = Image.open(img_path).convert('L')  # Converter para tons de cinza
image = image.resize((28, 28))  # Redimensionar para 28x28 pixels
image_array = np.array(image).astype('float32') / 255.0  # Normalizar os valores para [0, 1]

# Ajustar o formato para (1, 28, 28, 1), necessário para o modelo
image_array = image_array.reshape(1, 28, 28, 1)

# Fazer a previsão
prediction = model.predict(image_array)

# Obter a classe com maior probabilidade
predicted_class = np.argmax(prediction)
print(f"A classe prevista para a imagem é: {predicted_class}")


-----------------------------------------------------------------------------------------

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 28, 28, 32)          │             832 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 28, 28, 64)          │          51,264 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 14, 14, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 14, 14, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 12544)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │       1,605,760 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,659,146 (6.33 MB)
 Trainable params: 1,659,146 (6.33 MB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 401s 266ms/step - accuracy: 0.8813 - loss: 0.3744 - val_accuracy: 0.9853 - val_loss: 0.0502 - learning_rate: 0.0010
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 389s 259ms/step - accuracy: 0.9748 - loss: 0.0824 - val_accuracy: 0.9878 - val_loss: 0.0434 - learning_rate: 0.0010
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 382s 254ms/step - accuracy: 0.9824 - loss: 0.0576 - val_accuracy: 0.9902 - val_loss: 0.0354 - learning_rate: 0.0010
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 381s 254ms/step - accuracy: 0.9873 - loss: 0.0418 - val_accuracy: 0.9893 - val_loss: 0.0407 - learning_rate: 0.0010
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 386s 257ms/step - accuracy: 0.9873 - loss: 0.0424 - val_accuracy: 0.9894 - val_loss: 0.0372 - learning_rate: 0.0010
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 457s 267ms/step - accuracy: 0.9901 - loss: 0.0327 - val_accuracy: 0.9918 - val_loss: 0.0385 - learning_rate: 0.0010
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 424s 255ms/step - accuracy: 0.9903 - loss: 0.0312 - val_accuracy: 0.9904 - val_loss: 0.0396 - learning_rate: 0.0010
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 442s 255ms/step - accuracy: 0.9922 - loss: 0.0255 - val_accuracy: 0.9912 - val_loss: 0.0376 - learning_rate: 0.0010
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 0s 241ms/step - accuracy: 0.9927 - loss: 0.0222
Epoch 9: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 385s 257ms/step - accuracy: 0.9927 - loss: 0.0222 - val_accuracy: 0.9918 - val_loss: 0.0388 - learning_rate: 0.0010
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 439s 255ms/step - accuracy: 0.9944 - loss: 0.0163 - val_accuracy: 0.9925 - val_loss: 0.0381 - learning_rate: 5.0000e-04
