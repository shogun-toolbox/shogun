

from numpy import *
import numpy as np
import sys
import math as m
  
def  generate_circle_data():
	"""
	generate_circle_data generates a dataset in the shape of 2 circles.In this particular example we have taken the radius of 
	the 2 circles to be 10 and 30.
	'number_of_points_for_circle1' and 'number_of_points_for_circle2' define the number of y 
	coordinates for each circle.
	'row_vector': decides the dimension of the input data.In this particular example it is taken as 2
	'circle1':the first circle
	'circle2':the second circle
	'xmin_circle1':the minimum value of the x coordinate  for first circle
	'xmax_circle1':the maximum value of the x coordinate  for first circle
	'xmin_circle2':the minimum value of the x coordinate  for second circle
	'xmax_circle2':the maximum value of the x coordinate  for second circle
	'radius1':the radius of the first circle
	'radius2':the radius of the second circle
	"""
	number_of_points_for_circle1=42
	number_of_points_for_circle2=122
	row_vector=2
	circle1=[[0 for x in xrange(number_of_points_for_circle1)] for x in xrange(row_vector)] 
	circle2=[[0 for x in xrange(number_of_points_for_circle2)] for x in xrange(row_vector)] 
	indexi=0
	indexj=0
	l2=len(circle2[0])
	l1=len(circle1[0])
	xmin_circle1=-10
	xmax_circle1=10
	xmin_circle2=-30
	xmax_circle2=30
	for i in range(xmin_circle1,xmax_circle1+1):	
		circle1[0][indexi]=i
		circle1[0][indexi+(l1/2)]=i
		indexi=indexi+1
	for j in range(xmin_circle2,xmax_circle2+1):	
		circle2[0][indexj]=j
		circle2[0][indexj+(l2/2)]=j
		indexj=indexj+1

	indexi=0
	indexj=0
	radius1=10
	radius2=30
	i=radius1
	while (i>=0):
		y=(radius1*radius1) - (i*i)
		y2=m.sqrt(y)
		circle1[1][indexi]=y2
		circle1[1][indexi+(l1/2)]=-1*y2
		i=i-1
		indexi=indexi+1
	i=1
	while(i<radius1):
		y=(radius1*radius1) - (i*i)
		y2=m.sqrt(y)
		circle1[1][indexi]=y2
		circle1[1][indexi+(l1/2)]=-1*y2
		indexi=indexi+1
		i=i+1
	
	
	indexi=0
	indexj=0
	i=radius2
	while (i>=0):
		y=(radius2*radius2) - (i*i)
		y2=m.sqrt(y)
		circle2[1][indexi]=y2
		circle2[1][indexi+(l2/2)]=-1*y2
		i=i-1
		indexi=indexi+1
	i=1
	while(i<radius2):
		y=(radius2*radius2) - (i*i)
		y2=m.sqrt(y)
		circle2[1][indexi]=y2
		circle2[1][indexi+(l2/2)]=-1*y2
		indexi=indexi+1
		i=i+1	
	two_circles=hstack((circle1,circle2))
	return two_circles

if __name__=='__main__':
	data=generate_circle_data()
