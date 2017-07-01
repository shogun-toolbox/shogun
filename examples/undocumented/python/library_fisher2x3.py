#!/usr/bin/env python
from numpy import *
from modshogun import *

x=array([[20.0,15,15],[10,20,20]])
y=array([[21.0,21,18],[19,19,22]])
z=array([[15.0,27,18],[32,5,23]])


parameter_list = [[x,concatenate((x,y,z),1)]]

def library_fisher2x3_modular (table, tables):
	pval=Statistics_fishers_exact_test_for_2x3_table(table)
	pvals=Statistics_fishers_exact_test_for_multiple_2x3_tables(tables)
	return (pval,pvals)

if __name__=='__main__':
	print('Fisher 2x3')
	library_fisher2x3_modular(*parameter_list[0])
