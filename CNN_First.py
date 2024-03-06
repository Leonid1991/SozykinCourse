## Cifar 10, тренировка
# Подключаем пакеты
import numpy as np
import matplotlib.pyplot as plt
# Необходимые функции и пакеты
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing import image
# from keras.utils import np_utils
from keras import utils
# Размер мини-выборки и эпох для обучения
batch_size, nb_epoch  = 200, 20
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
# Создаем нейронную сеть (padding='same' нужно сохранить тот же размер изображкения, поэтому добавляем автоматически 0)
model = Sequential()                                             # Создаем последовательную модель
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(img_rows, img_cols, img_channels), activation='relu'))  # Первый сверточный слой
model.add(Conv2D(16, (3, 3), activation='relu', padding='same')) # Второй сверточный слой
model.add(MaxPooling2D(pool_size=(2, 2)))                        # Первый слой подвыборки  
model.add(Dropout(0.25))                                         # Слой регуляризации Dropout 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) # Третий сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu'))                 # Четвертый сверточный слой
model.add(MaxPooling2D(pool_size=(2, 2)))                        # Второй слой подвыборки
model.add(Dropout(0.25))                                         # Слой регуляризации Dropout
model.add(Flatten())                                             # Слой преобразования данных в плоское (1D ?!)
model.add(Dense(256, activation='relu'))                         # Полносвязный слой для классификации
model.add(Dropout(0.5))                                          # Слой регуляризации Dropout
model.add(Dense(10, activation='softmax'))                       # Выходной полносвязный слой
# Печатаем информацию о сети
print(model.summary())
# Компилируем модель
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Обучаем нейронную сеть
myFirstNN = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate(x_test, y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
# визуализация данных
myFirstNN_dict = myFirstNN.history
acc_values = myFirstNN_dict['accuracy']
val_acc_values = myFirstNN_dict['val_accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Сохраняем обученную нейронную сеть
'''model_json = model.to_json()
json_file = open("cifar10_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("cifar10_model.h5")'''
# Проверка на тестовых данных
i = -1
x = x_test[i]
x = np.expand_dims(x, axis=0)
prediction = np.argmax(model.predict(x))
print("Рассмотренный случай, i = ", i)
print("Предсказание", classes[prediction])
print("Реальность", classes[np.argmax(y_test[i])])
