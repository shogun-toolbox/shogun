
%% load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('-mat', '../data/DynProg_example.dat')

%% set a number of defaults
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
use_orf = 1;
num_svms = 8;
use_long_transitions = 1;
threshold = 1000;
long_transition_max_len = 100000;
block.content_pred(end+1:num_svms,:) = deal(0);
viterbi_nbest = [1 0] ;

%% reshape the training parameters and additional information like
%% length constraints and transformation type and pass them to shogun
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:length(penalty_array)
  all_ids(j)            = penalty_array{j}.id;
  all_names{j}          = penalty_array{j}.name;
  all_limits(:,j)       = penalty_array{j}.limits;
  all_penalties(:,j)    = penalty_array{j}.penalties;
  if isempty(penalty_array{j}.transform)
    all_transform{j}    = 'linear';
  else
    all_transform{j}    = penalty_array{j}.transform;
  end
  all_min_values(j)     = penalty_array{j}.min_value;
  all_max_values(j)     = penalty_array{j}.max_value;
  all_use_cache(j)      = penalty_array{j}.use_cache;
  all_use_svm(j)        = penalty_array{j}.use_svm;
  all_do_calc(j)        = 1;
end

sg('set_plif_struct',int32(all_ids)-1,all_names, all_limits, all_penalties, all_transform,...
	all_min_values, all_max_values, int32(all_use_cache), int32(all_use_svm), int32(all_do_calc));

%% pass the data to shogun
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sg('init_dyn_prog', num_svms)
sg('set_lin_feat', block.seq, int32(block.all_pos-1), block.content_pred);
sg('set_model', model.transition_pointers, use_orf, int32(model.mod_words), int32(state_signals),int32(model.orf_info))
sg('set_feature_matrix', block.features)
sg('long_transition_settings', use_long_transitions, threshold, long_transition_max_len)

%% run the dynamic program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[path_scores, path, ppos]= sg('best_path_trans', model.p', model.q', int32(viterbi_nbest), seg_path, a_trans, loss);

