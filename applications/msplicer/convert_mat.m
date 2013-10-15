function convert_mat()

addpath ../matlab
fnames={'../matlab/msplicer_elegansWS120_gc=0_orf=0.mat', '../matlab/msplicer_elegansWS120_gc=0_orf=1.mat', '../matlab/msplicer_elegansWS120_gc=1_orf=0.mat', '../matlab/msplicer_elegansWS150_gc=0_orf=0.mat', '../matlab/msplicer_elegansWS160_gc=0_orf=0.mat', '../matlab/msplicer_elegansWS160_gc=1_orf=0.mat', '../matlab/msplicer_elegansWS160_gc=1_orf=1.mat'};

for i=1:length(fnames),
	clear L;
	L=load(fnames{i});
	targetname=[ './data', fnames{i}(10:end-3), 'dat' ];

	fid=fopen(targetname,'wb');

	fprintf(fid, '%%msplicer definition file version: 1.0\n\n');
	fprintf(fid, 'bins=%d\n', L.bins);
	fprintf(fid, 'dict_weights_intron=');
	write_mat(fid, L.dict_weights_train.intron);
	fprintf(fid, 'dict_weights_coding=');
	write_mat(fid, L.dict_weights_train.coding);
	fprintf(fid,'\n');

    % has to fit to the python code (order of array in plif.py)
    penids.acceptor = 0 ;
    penids.donor = 1 ;
    penids.first_coding_len = 2 ;
    penids.last_coding_len = 3 ;
    penids.coding_len = 4 ;
    penids.single_coding_len = 5 ;
    penids.intron_len = 6 ;

	if ~isempty(findstr(targetname,'_orf=1'))
		make_a_trans_orf(fid, L, penids);
	else
		make_a_trans_noorf(fid, L, penids);
	end

	%penalties
	fprintf(fid,'%%penalties\n');
	write_penalty(fid, 'penalty_acceptor', L.penalty.acceptor);
	write_penalty(fid, 'penalty_donor', L.penalty.donor);
	write_penalty(fid, 'penalty_coding_len', L.penalty.coding_len);
	write_penalty(fid, 'penalty_first_coding_len', L.penalty.first_coding_len);
	write_penalty(fid, 'penalty_last_coding_len', L.penalty.last_coding_len);
	write_penalty(fid, 'penalty_single_coding_len', L.penalty.single_coding_len);
	write_penalty(fid, 'penalty_intron_len', L.penalty.intron_len);
	write_penalty(fid, 'penalty_coding', L.penalty.coding);
	write_penalty(fid, 'penalty_coding2', L.penalty.coding2);
	write_penalty(fid, 'penalty_coding3', L.penalty.coding3);
	write_penalty(fid, 'penalty_coding4', L.penalty.coding4);
	write_penalty(fid, 'penalty_intron', L.penalty.intron);
	write_penalty(fid, 'penalty_intron2', L.penalty.intron2);
	write_penalty(fid, 'penalty_intron3', L.penalty.intron3);
	write_penalty(fid, 'penalty_intron4', L.penalty.intron4);
	write_penalty(fid, 'penalty_transitions', L.penalty.transitions);
	fprintf(fid,'\n');

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

function make_a_trans_orf(fid, L, penids)
	[A,p,q,info,penalties,orf_info]=gen_splice_model_orf(penids);
	write_model(fid, L, A, p, q, info, penalties, orf_info)

function make_a_trans_noorf(fid, L, penids)
	[A, p, q, info, penalties, orf_info]=gen_splice_model_noorf(penids);
	write_model(fid, L, A,p,q, info, penalties, orf_info)

function write_model(fid, L, A,p,q, info, penalties, orf_info)
	A(~isinf(A))=L.penalty.transitions.penalty;
	%idx=[];
	%fieldns=fieldnames(info);
	%for i=1:length(fieldns)
	%	if isequal(fieldns{i}, 'cnt')
	%		continue
	%	end
	%	idx=[idx getfield(info, fieldns{i})];
	%end
	%A=A(idx,idx);

	a_trans = zeros(3,sum(~isinf(A(:)))) ;
	k=0 ;
	for i=1:size(A,1)
	  idx = find(~isinf(A(i,:))) ;
	  val = A(i,idx) ;
	  a_trans(1,k+1:k+length(idx))=i-1 ;
	  a_trans(2,k+1:k+length(idx))=idx-1 ;
	  a_trans(3,k+1:k+length(idx))=val ;
	  k=k+length(idx) ;
	end ;
	a_trans=a_trans' ;
	[tmp,idx]=sort(a_trans(:,2)) ;
	a_trans = a_trans(idx,:)' ;

	fprintf(fid, 'msplicer_a_trans=');
	write_mat(fid, a_trans);
	fprintf(fid, 'msplicer_p=');
	p(isinf(p))=32768;
	write_mat(fid, p(:));
	fprintf(fid, 'msplicer_q=');
	q(isinf(q))=32768;
	write_mat(fid, q(:));
	fprintf(fid,'\n');

    % start-state: 0
    % exon-start-state: 1
    % donor-state: 2
    % acceptor-state: 3
    % exon-end-state: 4
    % stop-state: 5
    statedescr = zeros(1,info.cnt) ;
    statedescr(info.start) = 0 ;
    statedescr(info.atg) = 1 ;
    statedescr(info.don) = 2 ;
    statedescr(info.acc) = 3 ;
    statedescr(info.stop) = 4 ;
    statedescr(info.final) = 5 ;

	fprintf(fid, 'statedescr=');
	write_mat(fid, statedescr);
	fprintf(fid,'\n');

    plifidmat = penalties ;
    plifidmat(plifidmat==0)=-1 ;

	fprintf(fid, 'plifidmat=');
	write_mat(fid, plifidmat);
	fprintf(fid,'\n');

	fprintf(fid, 'orf_info=');
	write_mat(fid, orf_info);
	fprintf(fid,'\n');

    word_degree = [3,4,5,6] ;
    mod_words   = [1,1,1,1,1,1,1,1;
                   0,0,0,0,0,0,0,0] ;
    sign_words   = [1,1,1,1,1,1,1,1] ;

	fprintf(fid, 'word_degree=');
	write_mat(fid, word_degree);
	fprintf(fid,'\n');

	fprintf(fid, 'mod_words=');
	write_mat(fid, mod_words);
	fprintf(fid,'\n');

	fprintf(fid, 'sign_words=');
	write_mat(fid, sign_words);
	fprintf(fid,'\n');

    info

function write_penalty(fid, name, x)

	if isfield(x, 'boundaries')
		fprintf(fid, '%s_boundaries=', name);
		write_mat(fid, x.boundaries(:,1:(end-1)));
	else
		warning('boundaries field does not exist!')
	end
	fprintf(fid, '%s_penalty=', name);
	write_mat(fid, x.penalty');

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
