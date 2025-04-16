#3.1
#String
data = 'hello world'
print(data[0])      
print(len(data))    
print(data)         
#Number
value = 123.1
print(value)

value = 10
print(value)
#Boolean
a = True
b = False
print(a, b)
# Multiple Assignment
a, b, c = 1, 2, 3
print(a, b, c)
#No value
a = None
print(a)
#3.1.2
#If- then -else
value = 99
if value == 99:
    print('That is fast')
elif value > 200:
    print('That is too fast')
else:
    print('That is safe')
#For loop
for i in range(10):
    print(i)
#While loop
i = 0
while i < 10:
    print(i)
    i += 1
#3.1.3
#Tuple
a = (1, 2, 3)
print(a)
#List
mylist = [1, 2, 3]
print("Zeroth Value: %d" % mylist[0])
mylist.append(4)
print("List Length: %d" % len(mylist))

for value in mylist:
    print(value)
#Dictionary
mydict = {'a': 1, 'b': 2, 'c': 3}
print("A value: %d" % mydict['a'])
mydict['a'] = 11
print("A value: %d" % mydict['a'])
print("Keys: %s" % mydict.keys())
print("Values: %s" % mydict.values())

for key in mydict.keys():
    print(mydict[key])
#Function
def mysum(x, y):
    return x + y

result = mysum(1, 3)
print(result)
#3.2
#Create array
import numpy
mylist = [1, 2, 3]
myarray = numpy.array(mylist)
print(myarray)
print(myarray.shape)
#Access data
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)

print(myarray)
print(myarray.shape)
print("First row: %s" % myarray[0])
print("Last row: %s" % myarray[-1])
print("Specific row and col: %s" % myarray[0, 2])
print("Whole col: %s" % myarray[:, 2])
#Arithmetic
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
print("Addition: %s" % (myarray1 + myarray2))
print("Multiplication: %s" % (myarray1 * myarray2))
#3.3
#Line plot
import matplotlib.pyplot as plt
import numpy

myarray = numpy.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()
#Scatter plot
import matplotlib.pyplot as plt
import numpy

myarray = numpy.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()
#3.4
#Series
import pandas
myarray = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
print(myseries)

print(myseries[0])
print(myseries['a'])
#Dataframe
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

print("method 1:")
print("one column: %s" % mydataframe['one'])

print("method 2:")
print("one column: %s" % mydataframe.one)

