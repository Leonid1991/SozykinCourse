## Задача из https://www.asozykin.ru/courses/nnpython
import matplotlib
from matplotlib import pyplot as plt
# Необходимые функции
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
# В Keras встроены средства работы с популярными наборами данных
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# Среднее значение
mean = x_train.mean(axis=0)
# Стандартное отклонение
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
# Создаем нейронную сеть
n_epoch = 10
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
# Компилируем сеть
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())
# Обучаем нейронную сеть
history = model.fit(x_train, y_train, epochs=n_epoch, batch_size=1, verbose=2)
# Оценка точности работы сети
mse, mae = model.evaluate(x_test, y_test, verbose=2)
print("=========================================================")
print("==============  Оценка точности работы сети  ============")
print("Средняя абсолютная ошибка (тысяч долларов):", mae)
# Проверка на тестовых данных
i = 1
print("Рассмотренный случай, i = ",i)
pred = model.predict(x_test)
print("Предсказанная стоимость:", pred[i][0], ", правильная стоимость:", y_test[i])
# Accessing loss values and accuracy from the history object
train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)
# Plotting the loss curve
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()