import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from PIL import Image

# Wczytanie danych treningowych i testowych z zestawu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Przygotowanie danych treningowych
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Budowa modelu CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Kompilacja i trening modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Ścieżki do plików JPG
image_paths = [
    r'C:\Users\PC\Pictures\cyfry\zero.jpg',
    r'C:\Users\PC\Pictures\cyfry\jeden.jpg',
    r'C:\Users\PC\Pictures\cyfry\dwa.jpg',
    r'C:\Users\PC\Pictures\cyfry\trzy.jpg',
    r'C:\Users\PC\Pictures\cyfry\cztery.jpg',
    r'C:\Users\PC\Pictures\cyfry\piec.jpg',
    r'C:\Users\PC\Pictures\cyfry\szesc.jpg',
    r'C:\Users\PC\Pictures\cyfry\siedem.jpg',
    r'C:\Users\PC\Pictures\cyfry\osiem.jpg',
    r'C:\Users\PC\Pictures\cyfry\dziewiec.jpg',
    
]

# Przetwarzanie każdego obrazu
for img_path in image_paths:
    img = Image.open(img_path).convert('L')  # Konwersja do skali szarości
    img = img.resize((28, 28))  # Dostosowanie rozmiaru obrazu
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Wyświetlenie rozpoznanej cyfry
    print(f"Na zdjęciu {img_path} jest liczba:", predicted_class)