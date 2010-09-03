function convert_mat()

%load('ARTS-info.mat');
load('/home/sonne/stuff/ARTS-info.mat');
targetname='ARTS.dat';

fid=fopen(targetname,'wb');

fprintf(fid, '%%arts version: 1.0\n\n');

idx=find(alphas~=0);
data=[trainData.xPos,trainData.xNeg];
alphas=alphas(idx);
center=trainData.tssPosition;

svs1=data(par.select1+center, idx);
svs2=data(par.select2+center, idx);
svs3=data(par.select3+center, idx);

fprintf(fid, 'b=%e\n', b);
fprintf(fid, 'alphas=');
write_mat(fid, alphas);

fprintf(fid, 'num_kernels=%d\n', 3);

fprintf(fid, 'kernel_name1=%s\n', 'wdshift');
fprintf(fid, 'kernel_left1=%d\n', min(par.select1));
fprintf(fid, 'kernel_center1=%d\n', 0);
fprintf(fid, 'kernel_right1=%d\n', max(par.select1));
fprintf(fid, 'kernel_order1=%d\n', par.order1);
fprintf(fid, 'kernel_shift1=%d\n', par.shift1);
fprintf(fid, 'kernel_svs1=');
write_string(fid, svs1);
fprintf(fid,'\n');

fprintf(fid, 'kernel_name2=%s\n', 'spectrum');
fprintf(fid, 'kernel_left2=%d\n', min(par.select2));
fprintf(fid, 'kernel_center2=%d\n', 0);
fprintf(fid, 'kernel_right2=%d\n', max(par.select2));
fprintf(fid, 'kernel_order2=%d\n', par.wordLen2);
fprintf(fid, 'kernel_svs2=');
write_string(fid, svs2);
fprintf(fid,'\n');

fprintf(fid, 'kernel_name3=%s\n', 'spectrum');
fprintf(fid, 'kernel_left3=%d\n', min(par.select3));
fprintf(fid, 'kernel_center3=%d\n', 0);
fprintf(fid, 'kernel_right3=%d\n', max(par.select3));
fprintf(fid, 'kernel_order3=%d\n', par.wordLen3);
fprintf(fid, 'kernel_svs3=');
write_string(fid, svs3);
fprintf(fid,'\n');
fclose(fid);

system(sprintf('bzip2 -9 "%s"\n', targetname));

function write_string(fid, x)
	fprintf(fid, '[\n');
	for i=1:size(x,2),
		fprintf(fid, '%c', x(1:(size(x,1)-1),i));
		fprintf(fid, '%c\n', x(size(x,1),i));
	end
	fprintf(fid, ']\n');

function write_mat(fid, x)
	if size(x,1)==1,
		fprintf(fid, '[');
		fprintf(fid, '%e, ', x(1:(length(x)-1)));
		fprintf(fid, '%e', x(end));
	else
		fprintf(fid, '[');
		for i=1:size(x,2),
			fprintf(fid, '%e, ', x(1:(size(x,1)-1),i));

			if i<size(x,2)
				fprintf(fid, '%e;\n ', x(size(x,1),i));
			else
				fprintf(fid, '%e', x(size(x,1),i));
			end
		end
	end
	fprintf(fid, ']\n');
