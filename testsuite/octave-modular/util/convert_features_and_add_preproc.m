function y = convert_features_and_add_preproc(prefix)
	eval(sprintf(['global ', prefix, 'order']));
	eval(sprintf(['global ', prefix, 'gap']));
	eval(sprintf(['global ', prefix, 'reverse']));
	eval(sprintf(['global ', prefix, 'feature_type']));
	order=eval(sprintf([prefix, 'order']));
	gap=eval(sprintf([prefix, 'gap']));
	reverse=eval(sprintf([prefix, 'reverse']));
	feature_type=eval(sprintf([prefix, 'feature_type']));
	global feats_train;
	global feats_test;
	y=false;

	if isempty(order)
		return;
	end

	charfeat_train=feats_train;
	charfeat_test=feats_test;

	if strcmp(feature_type, 'Ulong')==1
		global StringUlongFeatures;
		global SortUlongString;
		feats_train=StringUlongFeatures(charfeat_train.get_alphabet());
		feats_test=StringUlongFeatures(charfeat_test.get_alphabet());
		preproc=SortUlongString();
	elseif strcmp(feature_type, 'Word')==1
		global StringWordFeatures;
		global SortWordString;
		feats_train=StringWordFeatures(charfeat_train.get_alphabet());
		feats_test=StringWordFeatures(charfeat_test.get_alphabet());
		preproc=SortWordString();
	else
		return;
	end

	feats_train.obtain_from_char(charfeat_train,
		order-1, order, gap, tobool(reverse));
	feats_test.obtain_from_char(charfeat_test,
		order-1, order, gap, tobool(reverse));

	preproc.init(feats_train);
	feats_train.add_preproc(preproc);
	feats_train.apply_preproc();
	feats_test.add_preproc(preproc);
	feats_test.apply_preproc();

	y=true;
