clear gf

switch 3
case 1
	fid=fopen('/home/neuro/sonne/bb_hmm_examples/dice/dice.txt','r');
	sequence=fread(fid, inf, 'char');
	fclose(fid);

	sequence=sequence(1:end-1);
	char(sequence)'

	gf('send_command','pseudo 1e-10');
	gf('send_command','new_hmm 1 6');
	gf('set_features','TRAIN',char(sequence),'CUBE');
	gf('send_command', 'convert TRAIN SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TRAIN STRING CHAR STRING WORD');
	gf('send_command', 'bw');
	[hmm_p,hmm_q,hmm_a,hmm_b]=gf('get_hmm');

	exp(hmm_p)
	exp(hmm_q)
	exp(hmm_a)
	exp(hmm_b)
	(histc(sequence,49:54)/length(sequence))'

case 2
	fid=fopen('/home/neuro/sonne/bb_hmm_examples/dice/dice.txt','r');
	sequence=fread(fid, inf, 'char');
	fclose(fid);

	sequence=sequence(1:end-1);

	gf('send_command','pseudo 1e-10');
	gf('send_command','new_hmm 3 6');
	gf('set_features','TRAIN',char(sequence),'CUBE');
	gf('send_command', 'convert TRAIN SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TRAIN STRING CHAR STRING WORD');
	gf('send_command', 'bw');
	[hmm_p,hmm_q,hmm_a,hmm_b]=gf('get_hmm');

	exp(hmm_p)
	exp(hmm_q)
	exp(hmm_a)
	exp(hmm_b)
	disp('Verteilung: Wuerfel 1:0.346528 0.186111 0.344444 0.109722 0.002778 0.010417');
	disp('Verteilung: Wuerfel 2:0.398977 0.231458 0.149616 0.006394 0.039642 0.173913');
	disp('Verteilung: Wuerfel 3:0.101786 0.172024 0.168452 0.261905 0.066667 0.229167');
	
	gf('set_features','TEST',char(sequence),'CUBE');
	gf('send_command', 'convert TEST SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TEST STRING CHAR STRING WORD');
	[path,lik]=gf('get_viterbi_path',0);
case 3
	fid=fopen('/home/neuro/sonne/bb_hmm_examples/dna/acc_train.neg','r');
	negacc=fread(fid, inf, 'char');
	fclose(fid);

	fid=fopen('/home/neuro/sonne/bb_hmm_examples/dna/acc_train.pos','r');
	posacc=fread(fid, inf, 'char');
	fclose(fid);

	negacc=reshape(negacc,11,5000);
	posacc=reshape(posacc,11,5000);

	takeidx=1:length(negacc);
	takeidx=1:1000;
	negacc=negacc(1:10,takeidx);
	posacc=posacc(1:10,takeidx);

	negacc=negacc';
	posacc=posacc';

	gf('send_command','pseudo 1e-10');
	gf('send_command','new_hmm 4 4');
	gf('set_features','TRAIN',char(posacc),'DNA');
	gf('send_command', 'convert TRAIN SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TRAIN STRING CHAR STRING WORD');
	gf('send_command', 'bw');
	gf('send_command','set_hmm_as POS')

	gf('send_command','pseudo 1e-10');
	gf('send_command','new_hmm 4 4');
	gf('set_features','TRAIN',char(negacc),'DNA');
	gf('send_command', 'convert TRAIN SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TRAIN STRING CHAR STRING WORD');
	gf('send_command', 'bw');
	gf('send_command','set_hmm_as NEG')

	testacc=[negacc;posacc];

	gf('set_features','TEST',char(testacc)','DNA');
	gf('send_command', 'convert TEST SIMPLE CHAR STRING CHAR');
	gf('send_command', 'convert TEST STRING CHAR STRING WORD');
	out=gf('hmm_classify');
end
