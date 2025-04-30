# shape
print(dataset.shape) #(150, 5)

# head
print(dataset.head(20)) 
#      sepal-length   sepal-width   petal-length   petal-width   class
# 0        5.1            3.5           1.4             0.2     Iris-setosa
# 1        4.9            3.0           1.4             0.2     Iris-setosa
# 2        4.7            3.2           1.3             0.2     Iris-setosa
# 3        4.6            3.1           1.5             0.2     Iris-setosa
# 4        5.0            3.6           1.4             0.2     Iris-setosa
# 5        5.4            3.9           1.7             0.4     Iris-setosa
# 6        4.6            3.4           1.4             0.3     Iris-setosa
# 7        5.0            3.4           1.5             0.2     Iris-setosa
# 8        4.4            2.9           1.4             0.2     Iris-setosa
# 9        4.9            3.1           1.5             0.1     Iris-setosa
# 10       5.4            3.7           1.5             0.2     Iris-setosa
# 11       4.8            3.4           1.6             0.2     Iris-setosa
# 12       4.8            3.0           1.4             0.1     Iris-setosa
# 13       4.3            3.0           1.1             0.1     Iris-setosa
# 14       5.8            4.0           1.2             0.2     Iris-setosa
# 15       5.7            4.4           1.5             0.4     Iris-setosa
# 16       5.4            3.9           1.3             0.4     Iris-setosa
# 17       5.1            3.5           1.4             0.3     Iris-setosa
# 18       5.7            3.8           1.7             0.3     Iris-setosa
# 19       5.1            3.8           1.5             0.3     Iris-setosa

# descriptions
print(dataset.describe())
#        sepal-length   sepal-width   petal-length   petal-width
# count   150.000000     150.000000     150.000000     150.000000
# mean     5.843333      3.054000       3.758667       1.198667
# std      0.828066      0.433594       1.764420       0.763161
# min      4.300000      2.000000       1.000000       0.100000
# 25%      5.100000      2.800000       1.600000       0.300000
# 50%      5.800000      3.000000       4.350000       1.300000
# 75%      6.400000      3.300000       5.100000       1.800000
# max      7.900000      4.400000       6.900000       2.500000

# class distribution
print(dataset.groupby('class').size())
# class
# Iris-setosa     50
# Iris-versicolor 50
# Iris-virginica  50
