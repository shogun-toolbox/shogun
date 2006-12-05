load /home/schnarch/sonne/datasets/schering-chem/data/all_in_one_file.mat

sigma2 = 5;

%train
data1 = valdat(:,1:1000);
data2 = data1;

dpttrain = exp(-(repmat(diag(data1'*data1),[1 size(data2,2)]) + ...
      repmat(diag(data2'*data2)', [size(data1,2) 1]) - ...
	        2*data1'*data2)/sigma2);

data1 = valdat(:,1:1000);
data2 = valdat(:,1001:2000);

dpttest = exp(-(repmat(diag(data1'*data1),[1 size(data2,2)]) + ...
      repmat(diag(data2'*data2)', [size(data1,2) 1]) - ...
	        2*data1'*data2)/sigma2);

sg('set_features','TRAIN', data1);
sg('set_features','TEST', data2);
sg('send_command', 'convert_to_sparse TRAIN');
sg('send_command', 'convert_to_sparse TEST');


sg('send_command', 'set_kernel GAUSSIAN SPARSEREAL 10 5');

sg('send_command', 'init_kernel TRAIN');
dpttraingf=sg('get_kernel_matrix','TRAIN');

sg('send_command', 'init_kernel TEST');
dpttestgf=sg('get_kernel_matrix','TEST');

max(abs(dpttrain(:)-dpttrainsg(:)))
max(abs(dpttest(:)-dpttestsg(:)))
