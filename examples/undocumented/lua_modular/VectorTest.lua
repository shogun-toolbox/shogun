require 'shogun'
x = Labels()
y = {1, 3, 5, 7}
x:set_labels(y)
z = x:get_labels()
for k, v in pairs(z) do print (v) end
