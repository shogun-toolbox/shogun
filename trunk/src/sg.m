function [o1,o2,o3,o4]=sg(i1,i2) ;
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
%
%
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
