km=[[1,2,3];[4,5,6];[7,8,9]];
pairs=[[0,0]; [0,1]; [0,2]; [1,0]; [1,1]; [1,2]; [2,0]; [2,1]; [2,2]]';

%sg('loglevel', 'ALL');
sg('set_features', 'TEST', int32(pairs));
sg('set_kernel', 'TPPK', 'INT', 10, km);
x=sg('get_kernel_matrix', 'TEST');

x_ref=[];
for i=1:size(pairs,2),
	for j=1:size(pairs,2),
		a=pairs(1,i)+1;
		b=pairs(2,i)+1;
		c=pairs(1,j)+1;
		d=pairs(2,j)+1;
		x_ref(i,j)=km(a,c)*km(b,d)+km(a,d)*km(b,c);
	end
end

x
x_ref
