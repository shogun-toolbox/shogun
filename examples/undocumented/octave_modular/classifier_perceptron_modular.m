modshogun

num=50;
label_train_twoclass=[-ones(1,num/2) ones(1,num/2)];
fm_train_real=[randn(5,num/2)-1, randn(5,num/2)+1];
fm_test_real=[randn(5,num)-1, randn(5,num)+1];

% perceptron
disp('Perceptron')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

learn_rate=1.;
max_iter=1000;
num_threads=1;
labels=BinaryLabels(label_train_twoclass);

perceptron=Perceptron(feats_train, labels);
perceptron.set_learn_rate(learn_rate);
perceptron.set_max_iter(max_iter);
perceptron.parallel.set_num_threads(num_threads);
perceptron.train();

perceptron.set_features(feats_test);
perceptron.apply().get_labels();

