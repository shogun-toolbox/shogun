# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

x=array([[20.0,15,15],[10,20,20]])
y=array([[21,21,18],[19,19,22]])
z=array([[15,27,18],[32,5,23]])


parameter_list = [[x,concatenate((x,y,z),1)],[y,concatenate((y,y,x),1)]]

def library_fisher2x3_modular(table, tables)
# *** 	pval=Math_fishers_exact_test_for_2x3_table(table)
	pval=Modshogun::Math_fishers_exact_test_for_2x3_table.new
	pval.set_features(table)
# *** 	pvals=Math_fishers_exact_test_for_multiple_2x3_tables(tables)
	pvals=Modshogun::Math_fishers_exact_test_for_multiple_2x3_tables.new
	pvals.set_features(tables)
	return (pval,pvals)


end
if __FILE__ == $0
	puts 'Fisher 2x3'
	library_fisher2x3_modular(*parameter_list[0])

end
