import numpy as np
import matplotlib.pyplot as plt
import sys
import math as m
circle1=[[0 for x in xrange(42)] for x in xrange(2)] 
circle2=[[0 for x in xrange(122)] for x in xrange(2)] 
indexi=0
indexj=0
l2=len(circle2[0])
l1=len(circle1[0])
for i in range(-10,11):  
	circle1[0][indexi]=i
	circle1[0][indexi+(l1/2)]=i
	indexi=indexi+1
for j in range(-30,31):	
	circle2[0][indexj]=j
	circle2[0][indexj+(l2/2)]=j
	indexj=indexj+1
epsilon=float(0.98)
i=10
indexi=0
indexj=0
while (i>=0):
	y=100 - (i*i)
	y2=m.sqrt(y)
	circle1[1][indexi]=y2
	circle1[1][indexi+(l1/2)]=-1*y2
	i=i-1
	indexi=indexi+1
i=1
while(i<10):
	y=100 - (i*i)
	y2=m.sqrt(y)
	circle1[1][indexi]=y2
	circle1[1][indexi+(l1/2)]=-1*y2
	indexi=indexi+1
	i=i+1
	
#now plotting the points for circle 2
indexi=0
indexj=0
i=30
while (i>=0):
	y=900 - (i*i)
	y2=m.sqrt(y)
	circle2[1][indexi]=y2
	circle2[1][indexi+(l2/2)]=-1*y2
	i=i-1
	indexi=indexi+1
i=1
while(i<30):
	y=900 - (i*i)
	y2=m.sqrt(y)
	circle2[1][indexi]=y2
	circle2[1][indexi+(l2/2)]=-1*y2
	indexi=indexi+1
	i=i+1	
#plt.plot(circle1[1][:],circle1[0][:],'x',circle2[1][:],circle2[0][:],'o')
#plt.title('circle 1 and 2')
#plt.show()
