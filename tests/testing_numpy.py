import numpy

a = numpy.array([[1], [2], [3]])
b = numpy.array([[4], [5], [6]])
c = a+b

print(c[2])

print(numpy.zeros((3, 1)))
u = numpy.array([[1], [2], [3]])
v = numpy.array([[4], [5], [6]])
a = numpy.array([[7], [8], [9]])
s = numpy.array([[10.1], [12], [13]])
delta_time = 5
u = v
v = u + a*delta_time
s += u*delta_time + 0.5*a*delta_time*delta_time


print("done")
