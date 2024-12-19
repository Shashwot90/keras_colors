import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape
import numpy as np
import pandas

np.random.seed(10)
print (tf.__version__)

data = pandas.read_csv('colors.csv')