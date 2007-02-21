k = what('/.amd_mnt/huangho/export/kwaid0/home/jonas/shogun/trunk/python/testcases/mfiles');
res = [];
names = {};
for j=1:size(k.m)
  names{j}= k.m{j}
  res(j) = test_kernels(k.m{j});
end

for l=1:length(names)
 fprintf(1, [names{l}, '\t result: \t ',num2str(res(l)), '\n' ])
end



%res = test_kernels(k.m{1})