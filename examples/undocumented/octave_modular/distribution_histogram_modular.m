modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');

% Histogram
disp('Histogram')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_train_dna);
feats=StringWordFeatures(charfeat.get_alphabet());
feats.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats);
feats.add_preprocessor(preproc);
feats.apply_preprocessor();

histo=Histogram(feats);
histo.train();

histo.get_histogram();

num_examples=feats.get_num_vectors();
num_param=histo.get_num_model_parameters();
% for i=0:(num_examples-1),
% 	for j=0:(num_param-1),
% 		histo.get_log_derivative(j, i);
% 	end
% end

histo.get_log_likelihood();
histo.get_log_likelihood_sample();
