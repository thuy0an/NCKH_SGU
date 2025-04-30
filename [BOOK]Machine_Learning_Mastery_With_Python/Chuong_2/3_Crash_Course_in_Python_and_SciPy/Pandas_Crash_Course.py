#3.4.1 Series
import pandas
myarray = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
print(myseries)

print(myseries[0])
print(myseries['a'])
#3.4.2 Dataframe
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

print("method 1:")
print("one column: %s" % mydataframe['one'])

print("method 2:")
print("one column: %s" % mydataframe.one)
