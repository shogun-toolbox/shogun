%{
 #include "structure/DynProg.h" 
%}

%rename(CDynProg) DynProg;

/* model related functions */
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p, INT N)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* q, INT N)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* a, INT M, INT N)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* a_trans, INT num_trans, INT N)};

/* content svm related setup functions */
%apply (INT* IN_ARRAY1, INT DIM1) {(DREAL* p_num_degrees, INT p_num_svms)};
%apply (INT* IN_ARRAY1, INT DIM1) {(DREAL* p_word_degree_array, INT num_elem)};
%apply (INT* IN_ARRAY1, INT DIM1) {(DREAL* p_cum_num_words_array, INT num_elem)};
%apply (INT* IN_ARRAY1, INT DIM1) {(DREAL* p_num_words_array, INT num_elem)};

/* best_path preparation functions */
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* seq, INT N, INT seq_len)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* pos, INT seq_len)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* orf_info, INT m, INT n)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* segment_sum_weights, INT num_states, INT seq_len)};
%apply (CHAR* IN_ARRAY1, INT DIM1) {(CHAR* genestr, INT genestr_len)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* dictionary_weights, INT dict_len, INT n)};

/* best_path result retrieval functions */
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** scores, INT* n)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** states, INT* m, INT* n)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** positions, INT* m, INT* n)};

%include "structure/DynProg.h" 
