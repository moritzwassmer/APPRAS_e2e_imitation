
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense # TODO

def main():
    model = keras.models.load_model('C:/Users/morit/OneDrive/UNI/Master/WS22/APP-RAS/Programming/myfolder')
    print(model)

if __name__ == "__main__":
    main()
