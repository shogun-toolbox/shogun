%{
 #include "structure/DynProg.h" 
%}

%rename(DynProg) CDynProg;

#ifdef HAVE_PYTHON

/* model related functions */
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p, INT N)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* q, INT N)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* a, INT M, INT N)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* a_trans, INT num_trans, INT N)};

/* content svm related setup functions */
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* p_num_degrees, INT p_num_svms)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* p_word_degree_array, INT num_elem)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* p_cum_num_words_array, INT num_elem)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* p_num_words_array, INT num_elem)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* p_mod_words_array, INT num_elem, INT num_columns)};
%apply (bool* IN_ARRAY1, INT DIM1) {(bool * p_sign_words_array, INT num_elem)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* p_string_words_array, INT num_elem)};



/* best_path_trans preparation functions */
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* seq, INT N, INT seq_len)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* pos, INT seq_len)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* orf_info, INT m, INT n)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* segment_sum_weights, INT num_states, INT seq_len)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT *plif_id_matrix, INT m, INT n)}
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT *plif_id_matrix, INT m, INT n)}
%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* genestr, INT genestr_len, INT genestr_num)};
%apply (CHAR* IN_ARRAY1, INT DIM1) {(CHAR* genestr, INT genestr_len)};

/* best_path_trans_deriv preparation functions */
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* my_state_seq, INT seq_len)}
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* my_pos_seq, INT seq_len)}


%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* dictionary_weights, INT dict_len, INT n)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL * segment_loss, INT num_segment_id1, INT num_segment_id2)}
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* segment_ids_mask, INT m, INT n)}

/* best_path result retrieval functions */
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** scores, INT* n)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(INT** states, INT* m, INT* n)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(INT** positions, INT* m, INT* n)};

/* best_path_trans_deriv result retrieval functions */
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** my_scores, INT* seq_len)}
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** my_losses, INT* seq_len)}
#endif

%include "structure/DynProg.h" 
