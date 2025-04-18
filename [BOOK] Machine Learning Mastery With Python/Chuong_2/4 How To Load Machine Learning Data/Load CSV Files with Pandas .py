#Tải CSV với Pandas
from pandas import read_csv

filename = 'pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

print(data.shape)

url = 'https://goo.gl/vhm1eU'

data = read_csv(url, names=names)

print(data.shape)
