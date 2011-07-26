require 'modshogun'
require 'narray'
x = Modshogun::Labels.new
y = [1, 3, 5, 7]
x.set_labels(y)
z = x.get_labels()
for i in 0..3
	puts z[i];
end
