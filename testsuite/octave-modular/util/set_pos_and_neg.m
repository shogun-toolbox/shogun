function y = set_pos_and_neg()
	global pos;
	global pos_clone;
	global neg;
	global neg_clone;
	global N;
	global M;
	global pseudo;
	global order;
	global gap;
	global reverse;
	global data_train;
	global data_test;
	global StringCharFeatures;
	global StringWordFeatures;
	global SortWordString;
	global CUBE;
	global HMM;
	y=true;

	charfeat=StringCharFeatures(CUBE);
	charfeat.set_string_features(data_train);
	wordfeats_train=StringWordFeatures(charfeat.get_alphabet());
	wordfeats_train.obtain_from_char(charfeat,
		order-1, order, gap, tobool(reverse));
	preproc=SortWordString();
	preproc.init(wordfeats_train);
	wordfeats_train.add_preproc(preproc);
	wordfeats_train.apply_preproc();

	charfeat=StringCharFeatures(CUBE);
	charfeat.set_string_features(data_test);
	wordfeats_test=StringWordFeatures(charfeat.get_alphabet());
	wordfeats_test.obtain_from_char(charfeat,
		order-1, order, gap, tobool(reverse));
	wordfeats_test.add_preproc(preproc);
	wordfeats_test.apply_preproc();

	% cheating, BW_NORMAL is somehow not available
	BW_NORMAL=0;
	pos=HMM(wordfeats_train, N, M, pseudo);
	pos.train();
	pos.baum_welch_viterbi_train(BW_NORMAL);
	neg=HMM(wordfeats_train, N, M, pseudo);
	neg.train();
	neg.baum_welch_viterbi_train(BW_NORMAL);
	pos_clone=HMM(pos);
	neg_clone=HMM(neg);
	pos_clone.set_observations(wordfeats_test);
	neg_clone.set_observations(wordfeats_test);
