# Correction Matrix Plot 
from matplotlib import pyplot 
from pandas import read_csv 
import numpy 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) 
correlations = data.corr() 
# plot correlation matrix 
fig = pyplot.figure() 
ax = fig.add_subplot(111) 
cax = ax.matshow(correlations, vmin=-1, vmax=1) 
fig.colorbar(cax) 
ticks = numpy.arange(0,9,1) 
ax.set_xticks(ticks) 
ax.set_yticks(ticks) 
ax.set_xticklabels(names) 
ax.set_yticklabels(names) 
pyplot.show()
# Correction Matrix Plot (generic) 
from matplotlib import pyplot 
from pandas import read_csv 
import numpy 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) 
correlations = data.corr() 
# plot correlation matrix 
fig = pyplot.figure() 
ax = fig.add_subplot(111) 
cax = ax.matshow(correlations, vmin=-1, vmax=1) 
fig.colorbar(cax) 
pyplot.show()
# Scatterplot Matrix 
from matplotlib import pyplot 
from pandas import read_csv 
from pandas.tools.plotting import scatter_matrix 
filename = "pima-indians-diabetes.data.csv" 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) 
scatter_matrix(data) 
pyplot.show()
