import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape
import numpy as np
import pandas
import scipy.stats as stats
import pylab as plt

np.random.seed(10)
print (tf.__version__)

data = pandas.read_csv('colors.csv')
 
data.head()
len(data)
 
print(data)
names = data["name"]
h = sorted(names.str.len().as_matrix())
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
plt.plot(h,fit,'-o')
plt.hist(h,normed=True)
plt.xlabel('Chars')
plt.ylabel('Probability density')
plt.show()
np.array(h).max()
maxlen = 25
t = Tokenizer(char_level=True)
t.fit_on_texts(names)
tokenized = t.texts_to_sequences(names)
