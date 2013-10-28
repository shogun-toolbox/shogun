from numpy import *
import numpy as np
import sys
import math as m

class circle_data:
	def __init__(self):
		pass

	def  generate_data(self,number_of_points_for_circle1,number_of_points_for_circle2,row_vector):
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
		'xrange_circle1': the horizontal range of the x ccordinates of first circle
		'xrange_circle2': the horizontal range of the x ccordinates of second circle
		'radius1':the radius of the first circle
		'radius2':the radius of the second circle
		'two_circles':the concatenated data of the 2 circles
		"""
		#number_of_points_for_circle1=42
		#number_of_points_for_circle1=122
		row_vector=2
		circle1=zeros((row_vector,number_of_points_for_circle1))
		circle2=zeros((row_vector,number_of_points_for_circle2))
		radius1=(number_of_points_for_circle1-2)/4
		radius2=(number_of_points_for_circle2-2)/4
		l2=len(circle2[0])
		l1=len(circle1[0])
		xmin_circle1=-1*radius1
		xmax_circle1=radius1
		xmin_circle2=-1*radius2
		xmax_circle2=radius2
		xrange_circle1=range(xmin_circle1,xmax_circle1+1)
		xrange_circle1=xrange_circle1+xrange_circle1
		xrange_circle2=range(xmin_circle2,xmax_circle2+1)
		xrange_circle2=xrange_circle2+xrange_circle2
		circle1[0][:]=xrange_circle1
		circle2[0][:]=xrange_circle2
		mat1=ones((1,number_of_points_for_circle1))
		mat2=ones((1,number_of_points_for_circle2))
		mat1=radius1*radius1*ones((1,number_of_points_for_circle1))
		mat2=radius2*radius2*ones((1,number_of_points_for_circle2))
		circle1[1][:]=mat1-(circle1[0][:]*circle1[0][:])
		circle2[1][:]=mat2-(circle2[0][:]*circle2[0][:])

		circle1[1][:]=[m.sqrt(circle1[1][i]) for i in range(0,number_of_points_for_circle1)]
		circle1[1][(number_of_points_for_circle1/2):]=-1*circle1[1][(number_of_points_for_circle1/2):]

		circle2[1][:]=[m.sqrt(circle2[1][i]) for i in range(0,number_of_points_for_circle2)]
		circle2[1][(number_of_points_for_circle2/2):]=-1*circle2[1][(number_of_points_for_circle2/2):]

		two_circles=hstack((circle1,circle2))

		return two_circles
