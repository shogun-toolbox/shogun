function [o1,o2,o3,o4]=gf(i1,i2) ;
% 
%  gf('send_command', 'cmdline');
%
%  [p,q,a,b]          = gf('get_hmm');
%  [b,alpha]          = gf('get_svm');
%  [parms]            = gf('get_kernel_init');
%  [feature_matrix]   = gf('get_features', 'TRAIN|TEST');
%  [labels]           = gf('get_labels', 'TRAIN|TEST');
%  [parms]            = gf('get_preproc_init');
%  [p,q,a,b]          = gf('get_hmm_defs', 'cmdline');
%  [emm_prob, modsiz] = gf('get_plugin_estimate');
%            
%  gf('set_hmm', p,q,a,b);
%  gf('set_svm', b,alpha);
%  gf('set_kernel_init', parms);
%  gf('set_features', 'TRAIN|TEST', feature_matrix);
%  gf('set_labels', 'TRAIN|TEST', labels);
%  gf('set_preproc_init', parms);
%  gf('set_hmm_defs', p,q,a,b);
%  gf('set_plugin_estimate', emmission_prob, model_sizes);

%  out = gf('hmm_classify');
%  out = gf('hmm_classify_example');
%  out = gf('svm_classify');
%  out = gf('svm_classify_example');
%  out = gf('plugin_estimate_classify');
%  out = gf('plugin_estimate_classify_example');
