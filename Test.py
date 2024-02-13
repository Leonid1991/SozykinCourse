
## Cifar 10, тренировка
# Подключаем пакеты
import numpy as np
import matplotlib.pyplot as plt
# Необходимые функции и пакеты
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
from keras import utils
# Размер мини-выборки и эпох для обучения
batch_size, nb_epoch  = 100, 10
# Размер изображений (3 канала в изображении: RGB)
img_channels, img_rows, img_cols = 3, 32, 32
# Названия классов из набора данных CIFAR-10
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
# В Keras встроены средства работы с популярными наборами данных (загрузка данных)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''# Демонстрация картинки
n = 1
plt.imshow(X_train[n])
plt.show()
print("Номер класса:", y_train[n])
print("Тип объекта:", classes[y_train[n][0]])'''
# Нормализация данных 
x_train, x_test = x_train / 255, x_test / 255
# Разделяем ответы на разные категории (10 - количество классов изображений)
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10) 

print(x_test.shape)