require 'shogun'
x = RealFeatures()
y = {{1, 2, 3}, {4, 5, 6}}
x:set_feature_matrix(y)
z = x:get_feature_matrix();
for _,row in ipairs(z) do for _,cell in ipairs(row) do io.write(cell..' ') end io.write('\n') end

