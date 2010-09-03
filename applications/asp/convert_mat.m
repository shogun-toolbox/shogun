function convert_mat()

addpath /fml/ag-raetsch/home/sonne/svn/projects/poim/experiments
which_kernel=1; %index of WD kernel

splice_types={'acc','don'};
organisms={'H_sapiens_nohomologs', 'A_thaliana_paper', 'D_melanogaster_old', 'D_rerio', 'C_elegans'};
splice_aliases={'Acceptor','Donor'};
organism_aliases={'Human','Cress', 'Fly', 'Fish', 'Worm'};

for org_idx=1:length(organisms),

	targetname=[ './data/', organism_aliases{org_idx}, '.dat' ];
	fid=fopen(targetname,'wb');
	fprintf(fid, '%%asplicer definition file version: 1.0\n\n');

	%Acceptor
	type_idx=1;
	svmFileName=sprintf('/fml/ag-raetsch/share/projects/genefinding/%s/sensors/%s/output/SVMs%s_partition=1.mat', organisms{org_idx}, splice_types{type_idx}, splice_types{type_idx})
	if ~exist(svmFileName, 'file'),
		svmFileName=sprintf('/fml/ag-raetsch/share/projects/genefinding/%s/sensors/%s/output_SVMWD/SVM_best_partition=1.mat', organisms{org_idx}, splice_types{type_idx});
	end

	acc=load(svmFileName);
	range=acc.Train.PAR.method.center+[acc.Train.PAR.method.kernels{which_kernel}.lwin:acc.Train.PAR.method.kernels{which_kernel}.rwin];
	fprintf(fid,'%%acceptor splice\n');
	fprintf(fid, 'acc_splice_b=%e\n', acc.Train.b);
	fprintf(fid, 'acc_splice_order=%d\n', acc.Train.PAR.method.kernels{which_kernel}.order);
	fprintf(fid, 'acc_splice_window_left=%d\n', abs(acc.Train.PAR.method.kernels{which_kernel}.lwin));
	fprintf(fid, 'acc_splice_window_right=%d\n', abs(acc.Train.PAR.method.kernels{which_kernel}.rwin-1));
	fprintf(fid, 'acc_splice_alphas=');
	write_mat(fid, acc.Train.alphas(:,1));
	fprintf(fid, 'acc_splice_svs=');
	write_string(fid, acc.Train.XT(range,:));
	fprintf(fid,'\n');
	clear acc

	%Donor
	type_idx=2;
	svmFileName=sprintf('/fml/ag-raetsch/share/projects/genefinding/%s/sensors/%s/output/SVMs%s_partition=1.mat', organisms{org_idx}, splice_types{type_idx}, splice_types{type_idx})
	if ~exist(svmFileName, 'file'),
		svmFileName=sprintf('/fml/ag-raetsch/share/projects/genefinding/%s/sensors/%s/output_SVMWD/SVM_best_partition=1.mat', organisms{org_idx}, splice_types{type_idx});
	end

	don=load(svmFileName);
	range=don.Train.PAR.method.center+[don.Train.PAR.method.kernels{which_kernel}.lwin:don.Train.PAR.method.kernels{which_kernel}.rwin];
	fprintf(fid,'%%doneptor splice\n');
	fprintf(fid, 'don_splice_b=%e\n', don.Train.b);
	fprintf(fid, 'don_splice_order=%d\n', don.Train.PAR.method.kernels{which_kernel}.order);
	fprintf(fid, 'don_splice_window_left=%d\n', abs(don.Train.PAR.method.kernels{which_kernel}.lwin));
	fprintf(fid, 'don_splice_window_right=%d\n', abs(don.Train.PAR.method.kernels{which_kernel}.rwin-1));
	fprintf(fid, 'don_splice_alphas=');
	write_mat(fid, don.Train.alphas(:,1));
	fprintf(fid, 'don_splice_svs=');
	write_string(fid, don.Train.XT(range,:));
	fprintf(fid,'\n');
	clear don

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
