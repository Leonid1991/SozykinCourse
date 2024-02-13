## Задача из https://www.asozykin.ru/courses/nnpython
# Подключаем пакеты
import tensorflow as tf 
import numpy as np
# Необходимые функции
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import utils
# В Keras встроены средства работы с популярными наборами данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Преобразование размерности
x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)
# Применяются к каждому элементу массива отдельно
x_train, x_test = x_train / 255, x_test / 255
# Разделяем ответы на разные категории 
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10) 
# Создаем нейронную сеть
model = Sequential()                                    # Создаем последовательную модель
model.add(Dense(800, input_dim=784, activation="relu")) # Входной полносвязный слой, 800 нейронов, 784 входа в каждый нейрон
model.add(Dense(10, activation="softmax"))              # Выходной полносвязный слой, 10 нейронов (по количеству рукописных цифр)
# Компилируем сеть
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())
# Обучаем нейронную сеть
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)
# Проверка на тестовых данных
prediction = model.predict(x_test) # подстановка тестовых данных
i = -1
print("Рассмотренный случай, i = ",i)
print("Предсказание", np.argmax(prediction[i]))
print("Реальность", np.argmax(y_test[i]))
# Сохранение сети
model.save('FashionMnist.h5') # .keras тоже хорошо