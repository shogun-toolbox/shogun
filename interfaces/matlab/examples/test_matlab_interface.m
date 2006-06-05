%gf('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TRAIN 0');
%gf('send_command', 'reshape TRAIN 120 5000');
%features=gf('get_features','TRAIN');
%features(1)=17;
%features(2)=18;
%features(23)=23;
%gf('set_features','TRAIN', features);
%features2=gf('get_features','TRAIN');
%
%gf('send_command', 'load_labels /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst TRAIN');
%labels=gf('get_labels','TRAIN');
%labels(1)=17;
%labels(2)=18;
%labels(23)=23;
%gf('set_labels','TRAIN', labels);
%labels2=gf('get_labels','TRAIN');
%for i=1:1000

%gf('send_command', 'new_hmm 10 4 1');
%[p1,q1,a1,b1]=gf('get_hmm');
%gf('send_command', 'save_hmm bla.m');
%bla
%p2=p1-p
%q2=q1-q
%a2=a1-a
%b2=b1-b
%gf('set_hmm', p2,q2,a2,b2);
%[p3,q3,a3,b3]=gf('get_hmm');
%gf('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TRAIN 0');
%gf('send_command', 'reshape TRAIN 120 5000');
%fid=fopen('/home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawlab.tst', 'rb')
%lab=fread(fid,inf,'int');
%lab=double(lab');
%fclose(fid);
%gf('set_labels', 'TRAIN', lab);
%gf('set_labels', 'TEST', lab);
%%gf('send_command', 'load_labels /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawlab.tst TRAIN');
%gf('send_command', 'convert_to_sparse TRAIN');
%gf('send_command', 'set_kernel POLY SPARSEREAL 23 0');
%gf('send_command', 'init_kernel TRAIN');
%gf('send_command', 'new_svm LIGHT');
%gf('send_command', 'c 1');
%gf('send_command', 'svm_train');
%gf('send_command', 'save_svm /home/neuro/sonne/projects/work/first/research/genefinder/testdata/svm.ref');
%[b2, alphas2]=gf('get_svm');
%
%gf('send_command', 'load_features /home/neuro/sonne/projects/work/first/fraunhofer/schering-chem/data/gf_rawdat.tst REAL TEST 0');
%gf('send_command', 'reshape TEST 120 5000');
%gf('send_command', 'convert_to_sparse TEST');
%gf('send_command', 'svm_test /tmp/svm_out');
%[out(1)]=gf('svm_classify_example', 1);
%for i=0:4999,
%	out(i+1)=gf('svm_classify_example',i);
%end
%
%[out(1)]=gf('svm_classify_example', 4997);
%[lout]=gf('svm_classify');

gf('send_command', 'set_threshold 0');
gf('send_command', 'load_features /home/schnarch/sonne/home/data/acc_train_data.ascii CHAR TRAIN 0');
gf('send_command', 'load_labels /home/schnarch/sonne/home/data/acc_train_label.double TRAIN');
gf('send_command', 'load_features /home/schnarch/sonne/home/data/acc_test_data.ascii CHAR TEST 0');
gf('send_command', 'load_labels /home/schnarch/sonne/home/data/acc_test_label.double TEST');
gf('send_command', 'convert_char_to_word TRAIN DNA 3');
gf('send_command', 'convert_char_to_word TEST DNA 3');
gf('send_command', 'new_plugin_estimator');
gf('send_command', 'train_estimator');
gf('send_command', 'test_estimator /tmp/out2');

for i=0:39885,
	out(i+1)=gf('plugin_estimate_classify_example',i);
end
[lout]=gf('plugin_estimate_classify');
