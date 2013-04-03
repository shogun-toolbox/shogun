function y = set_pos_and_neg(prefix)
	eval(sprintf(['global ', prefix, 'N']));
	eval(sprintf(['global ', prefix, 'M']));
	eval(sprintf(['global ', prefix, 'pseudo']));
	eval(sprintf(['global ', prefix, 'order']));
	eval(sprintf(['global ', prefix, 'gap']));
	eval(sprintf(['global ', prefix, 'reverse']));
	eval(sprintf(['global ', prefix, 'data_train']));
	eval(sprintf(['global ', prefix, 'data_test']));
	global StringCharFeatures;
	global StringWordFeatures;
	global SortWordString;
	global CUBE;
	global HMM;
	global BW_NORMAL;
	global init_random;

	Math_init_random(init_random);
	N=eval(sprintf([prefix, 'N']));
	M=eval(sprintf([prefix, 'M']));
	pseudo=eval(sprintf([prefix, 'pseudo']));
	order=eval(sprintf([prefix, 'order']));
	gap=eval(sprintf([prefix, 'gap']));
	reverse=tobool(eval(sprintf([prefix, 'reverse'])));
	data_train=eval(sprintf([prefix, 'data_train']));
	data_test=eval(sprintf([prefix, 'data_test']));

	global pos_train;
	global pos_test;
	global neg_train;
	global neg_test;
	y=true;

	charfeat=StringCharFeatures(CUBE);
	charfeat.set_features(data_train);
	wordfeats_train=StringWordFeatures(charfeat.get_alphabet());
	wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
	preproc=SortWordString();
	preproc.init(wordfeats_train);
	wordfeats_train.add_preprocessor(preproc);
	wordfeats_train.apply_preprocessor();

	charfeat=StringCharFeatures(CUBE);
	charfeat.set_features(data_test);
	wordfeats_test=StringWordFeatures(charfeat.get_alphabet());
	wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
	wordfeats_test.add_preprocessor(preproc);
	wordfeats_test.apply_preprocessor();

	pos_train=HMM(wordfeats_train, N, M, pseudo);
	pos_train.train();
	pos_train.baum_welch_viterbi_train(BW_NORMAL);
	neg_train=HMM(wordfeats_train, N, M, pseudo);
	neg_train.train();
	neg_train.baum_welch_viterbi_train(BW_NORMAL);
	pos_test=HMM(pos_train);
	pos_test.set_observations(wordfeats_test);
	neg_test=HMM(neg_train);
	neg_test.set_observations(wordfeats_test);
