function [o1,o2,o3,o4]=sg(i1,i2)
% 
%  sg('send_command', 'cmdline');
%
%  [p,q,a,b]          = sg('get_hmm');
%  [b,alpha]          = sg('get_svm');
%  [parms]            = sg('get_kernel_init');
%  [feature_matrix]   = sg('get_features', 'TRAIN|TEST');
%  [labels]           = sg('get_labels', 'TRAIN|TEST');
%  [parms]            = sg('get_preproc_init');
%  [p,q,a,b]          = sg('get_hmm_defs', 'cmdline');
%  [emm_prob, modsiz] = sg('get_plugin_estimate');
%            
%  sg('set_hmm', p,q,a,b);
%  sg('set_svm', b,alpha);
%  sg('set_kernel_init', parms);
%  sg('add_features', 'TRAIN|TEST', feature_matrix);
%  sg('set_labels', 'TRAIN|TEST', labels);
%  sg('set_preproc_init', parms);
%  sg('set_hmm_defs', p,q,a,b);
%  sg('set_plugin_estimate', emmission_prob, model_sizes);
%
%  out = sg('hmm_classify');
%  out = sg('hmm_classify_example');
%  out = sg('one_class_hmm_classify');
%  out = sg('one_class_hmm_classify_example');
%  out = sg('one_class_linear_hmm_classify');
%  out = sg('svm_classify');
%  out = sg('svm_classify_example');
%  out = sg('plugin_estimate_classify');
%  out = sg('plugin_estimate_classify_example');
return
