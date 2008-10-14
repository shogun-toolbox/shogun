function convert_mat()

addpath ../matlab
fnames={'../matlab/x.mat'};

for i=1:length(fnames),
	clear L;
	L=load(fnames{i});
	targetname=[ './data', fnames{i}(10:end-3), 'dat' ];

	fid=fopen(targetname,'wb');

	fprintf(fid, '%%asplicer definition file version: 1.0\n\n');

	acc=load(L.accfname);
	fprintf(fid,'%%acceptor splice\n');
	fprintf(fid, 'acc_splice_b=%e\n', acc.b);
	fprintf(fid, 'acc_splice_order=%d\n', acc.PAR.order);
	fprintf(fid, 'acc_splice_window_left=%d\n', 60);
	fprintf(fid, 'acc_splice_window_right=%d\n', 79);
	fprintf(fid, 'acc_splice_alphas=');
	write_mat(fid, acc.alphas);
	fprintf(fid, 'acc_splice_svs=');
	write_string(fid, acc.XT);
	fprintf(fid,'\n');

	don=load(L.donfname);
	fprintf(fid,'%%donor splice\n');
	fprintf(fid, 'don_splice_b=%e\n', don.b);
	fprintf(fid, 'don_splice_use_gc=%d\n', don.PAR.use_gc); 
	fprintf(fid, 'don_splice_order=%d\n', don.PAR.order);
	fprintf(fid, 'don_splice_window_left=%d\n', 80);
	fprintf(fid, 'don_splice_window_right=%d\n', 59);
	fprintf(fid, 'don_splice_alphas=');
	write_mat(fid, don.alphas);
	fprintf(fid, 'don_splice_svs=');
	write_string(fid, don.XT);

	fclose(fid);

	system(sprintf('bzip2 -9 "%s"\n', targetname));
end

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
