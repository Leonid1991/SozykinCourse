## Задача из https://www.asozykin.ru/courses/nnpython
# Подключаем пакеты
import tensorflow as tf
import numpy as np 
# Необходимые функции
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras import utils
# В Keras встроены средства работы с популярными наборами данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Преобразование 
x_test = x_test.reshape(10000, 784)
# Применяются к каждому элементу массива отдельно
x_test = x_test / 255
# Разделяем ответы на разные категории 
y_test = utils.to_categorical(y_test, 10) 
# Загружаем готовую сеть 
model = load_model('FashionMnist.h5') 
# Проверка на тестовых данных
prediction = model.predict(x_test) # подстановка тестовых данных
i = -1
print("Рассмотренный случай, i = ",i)
print("Предсказание", np.argmax(prediction[i]))
print("Реальность", np.argmax(y_test[i]))
 
