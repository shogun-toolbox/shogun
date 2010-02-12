truth = sign(2*rand(1,60) - 1);
km=rand(length(truth));
km=km+km';

sg('set_kernel', 'CUSTOM', km, 'FULL');
sg('set_labels', 'TRAIN', truth);
sg('new_classifier', 'LIBSVM');
sg('train_classifier');
out_all = sg('classify');
out = sg('classify_example',0);
