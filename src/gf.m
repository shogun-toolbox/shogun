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
%  gf('add_features', 'TRAIN|TEST', feature_matrix);
%  gf('set_labels', 'TRAIN|TEST', labels);
%  gf('set_preproc_init', parms);
%  gf('set_hmm_defs', p,q,a,b);
%  gf('set_plugin_estimate', emmission_prob, model_sizes);

%  out = gf('hmm_classify');
%  out = gf('hmm_classify_example');
%  out = gf('one_class_hmm_classify');
%  out = gf('one_class_hmm_classify_example');
%  out = gf('one_class_linear_hmm_classify');
%  out = gf('svm_classify');
%  out = gf('svm_classify_example');
%  out = gf('plugin_estimate_classify');
%  out = gf('plugin_estimate_classify_example');


%%%%TODO comment these functions%%%%%
%else if (!strncmp(action, N_GET_KERNEL_OPTIMIZATION, strlen(N_GET_KERNEL_OPTIMIZATION)))
%else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
%else if (!strncmp(action, N_GET_KERNEL_INIT, strlen(N_GET_KERNEL_INIT)))
%else if (!strncmp(action, N_GET_PREPROC_INIT, strlen(N_GET_PREPROC_INIT)))
%else if (!strncmp(action, N_GET_HMM_DEFS, strlen(N_GET_HMM_DEFS)))
%else if (!strncmp(action, N_BEST_PATH_NO_B_TRANS, strlen(N_BEST_PATH_NO_B_TRANS)))
%else if (!strncmp(action, N_BEST_PATH_TRANS, strlen(N_BEST_PATH_TRANS)))
%else if (!strncmp(action, N_MODEL_PROB_NO_B_TRANS, strlen(N_MODEL_PROB_NO_B_TRANS)))
%else if (!strncmp(action, N_BEST_PATH_NO_B, strlen(N_BEST_PATH_NO_B)))
%else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
%				else if (!strncmp(target, "TEST", strlen("TEST")))
%else if (!strncmp(action, N_CLEAN_FEATURES, strlen(N_CLEAN_FEATURES)))
%				else if (!strncmp(target, "TEST", strlen("TEST")))
%else if (!strncmp(action, N_TRANSLATE_STRING, strlen(N_TRANSLATE_STRING)))
%else if (!strncmp(action, N_CRC, strlen(N_CRC)))
%else if (!strncmp(action, N_SET_PREPROC_INIT, strlen(N_SET_PREPROC_INIT)))
%else if (!strncmp(action, N_SET_HMM_DEFS, strlen(N_SET_HMM_DEFS)))
