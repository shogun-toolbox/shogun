%sg('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TRAIN 0');
%sg('send_command', 'reshape TRAIN 120 5000');
%features=sg('get_features','TRAIN');
%features(1)=17;
%features(2)=18;
%features(23)=23;
%sg('set_features','TRAIN', features);
%features2=sg('get_features','TRAIN');
%
%sg('send_command', 'load_labels /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst TRAIN');
%labels=sg('get_labels','TRAIN');
%labels(1)=17;
%labels(2)=18;
%labels(23)=23;
%sg('set_labels','TRAIN', labels);
%labels2=sg('get_labels','TRAIN');
%for i=1:1000

%sg('send_command', 'new_hmm 10 4 1');
%[p1,q1,a1,b1]=sg('get_hmm');
%sg('send_command', 'save_hmm bla.m');
%bla
%p2=p1-p
%q2=q1-q
%a2=a1-a
%b2=b1-b
%sg('set_hmm', p2,q2,a2,b2);
%[p3,q3,a3,b3]=sg('get_hmm');
%sg('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TRAIN 0');
%sg('send_command', 'reshape TRAIN 120 5000');
%fid=fopen('/home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawlab.tst', 'rb')
%lab=fread(fid,inf,'int');
%lab=double(lab');
%fclose(fid);
%sg('set_labels', 'TRAIN', lab);
%sg('set_labels', 'TEST', lab);
%%sg('send_command', 'load_labels /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawlab.tst TRAIN');
%sg('send_command', 'convert_to_sparse TRAIN');
%sg('send_command', 'set_kernel POLY SPARSEREAL 23 0');
%sg('send_command', 'init_kernel TRAIN');
%sg('send_command', 'new_svm LIGHT');
%sg('send_command', 'c 1');
%sg('send_command', 'svm_train');
%sg('send_command', 'save_svm /home/neuro/sonne/projects/work/first/research/genefinder/testdata/svm.ref');
%[b2, alphas2]=sg('get_svm');
%
%sg('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TEST 0');
%sg('send_command', 'reshape TEST 120 5000');
%sg('send_command', 'convert_to_sparse TEST');
%sg('send_command', 'svm_test /tmp/svm_out');
%[out(1)]=sg('svm_classify_example', 1);
%for i=0:4999,
%	out(i+1)=sg('svm_classify_example',i);
%end
%
%[out(1)]=sg('svm_classify_example', 4997);
%[lout]=sg('svm_classify');

sg('send_command', 'set_threshold 0');
sg('send_command', 'load_features /home/schnarch/sonne/home/data/acc_train_data.ascii CHAR TRAIN 0');
sg('send_command', 'load_labels /home/schnarch/sonne/home/data/acc_train_label.double TRAIN');
sg('send_command', 'load_features /home/schnarch/sonne/home/data/acc_test_data.ascii CHAR TEST 0');
sg('send_command', 'load_labels /home/schnarch/sonne/home/data/acc_test_label.double TEST');
sg('send_command', 'convert_char_to_word TRAIN DNA 3');
sg('send_command', 'convert_char_to_word TEST DNA 3');
sg('send_command', 'new_plugin_estimator');
sg('send_command', 'train_estimator');
sg('send_command', 'test_estimator /tmp/out2');

for i=0:39885,
	out(i+1)=sg('plugin_estimate_classify_example',i);
end
[lout]=sg('plugin_estimate_classify');
