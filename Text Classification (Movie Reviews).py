import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.chdir("C:/Users/Ahmar Ali Khan/Downloads/Talha/Tensor/Movie Review Classification")

data = keras.datasets.imdb

(train_data, train_labels) , (test_data, test_labels) = data.load_data(num_words = 100000)

#print(test_data[7])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["PAD"] = 0
word_index["START"] = 1
word_index["UNK"] = 2
word_index["UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["PAD"], padding = "post", maxlen = 350)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["PAD"], padding = "post", maxlen = 350)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[7]))
#print(len(test_data[32]), len(train_data[7]))

#model starts here

# model = keras.Sequential()

# model.add(keras.layers.Embedding(100000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation = "relu"))
# model.add(keras.layers.Dense(1, activation = "sigmoid"))

# model.summary()

# model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# x_val = train_data[:10000]
# x_train = train_data[10000:]

# y_val = train_labels[:10000]
# y_train = train_labels[10000:]

# model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

# results = model.evaluate(test_data, test_labels)

# print(results)

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

model = keras.models.load_model("model.h5")

with open("The Dark Knight Review.txt") as f:
    for line in f.readlines():
        nline = line.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode =  keras.preprocessing.sequence.pad_sequences([encode], value = word_index["PAD"], padding = "post", maxlen = 350)
        predict = model.predict(np.array(encode))
        pos_percent = predict[0][0] * 100
        pos_percent = float(pos_percent)
        print(line)
        print(encode)
        print("The Review is %.2f" %(pos_percent) + "% Positive.")


# model.save("model.h5")

# test_review = test_data[0]
# predict = model.predict(np.array([test_review]))
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction: " + str(predict))
# print("Actual: " + str(test_labels[0]))