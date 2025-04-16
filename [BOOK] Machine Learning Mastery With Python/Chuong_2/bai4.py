#Tải CSV với Thư Viện Chuẩn của Python
import csv
import numpy

filename = 'pima-indians-diabetes.data.csv'

raw_data = open(filename, 'rb')

reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)

data = numpy.array(x).astype('float')

print(data.shape)
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
#Tải CSV với Pandas
from pandas import read_csv

filename = 'pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

print(data.shape)

url = 'https://goo.gl/vhm1eU'

data = read_csv(url, names=names)

print(data.shape)
