# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

x=array([[20.0,15,15],[10,20,20]])
y=array([[21,21,18],[19,19,22]])
z=array([[15,27,18],[32,5,23]])


parameter_list = [[x,concatenate((x,y,z),1)],[y,concatenate((y,y,x),1)]]

def library_fisher2x3_modular(table, tables)
	pval=Math_fishers_exact_test_for_2x3_table(table)
	pvals=Math_fishers_exact_test_for_multiple_2x3_tables(tables)
	return (pval,pvals)


end
if __FILE__ == $0
	print 'Fisher 2x3'
	library_fisher2x3_modular(*parameter_list[0])

end
