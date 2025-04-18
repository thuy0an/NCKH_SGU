#Tải CSV với NumPy
from numpy import loadtxt

filename = 'pima-indians-diabetes.data.csv'

raw_data = open(filename, 'rb')

data = loadtxt(raw_data, delimiter=",")

print(data.shape)

from urllib import urlopen

url = 'https://goo.gl/vhm1eU'
raw_data = urlopen(url)

dataset = loadtxt(raw_data, delimiter=",")

print(dataset.shape)
