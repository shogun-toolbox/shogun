%{
 #include "structure/DynProg.h"
%}

%rename(DynProg) CDynProg;

#ifdef HAVE_PYTHON

/* model related functions */
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p, int32_t N)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* q, int32_t N)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* a, int32_t M, int32_t N)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* a_trans, int32_t num_trans, int32_t N)};

/* content svm related setup functions */
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* p_num_degrees, int32_t p_num_svms)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* p_word_degree_array, int32_t num_elem)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* p_cum_num_words_array, int32_t num_elem)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* p_num_words_array, int32_t num_elem)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* p_mod_words_array, int32_t num_elem, int32_t num_columns)};
%apply (bool* IN_ARRAY1, int32_t DIM1) {(bool * p_sign_words_array, int32_t num_elem)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* p_string_words_array, int32_t num_elem)};

/* best_path_trans preparation functions */
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* seq, int32_t N, int32_t seq_len)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* pos, int32_t seq_len)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* orf_info, int32_t m, int32_t n)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* segment_sum_weights, int32_t num_states, int32_t seq_len)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t *plif_id_matrix, int32_t m, int32_t n)}
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t *plif_id_matrix, int32_t m, int32_t n)}
%apply (char* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(char* genestr, int32_t genestr_len, int32_t genestr_num)};
%apply (char* IN_ARRAY1, int32_t DIM1) {(char* genestr, int32_t genestr_len)};

/* best_path_trans_deriv preparation functions */
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* my_state_seq, int32_t seq_len)}
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* my_pos_seq, int32_t seq_len)}


%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* dictionary_weights, int32_t dict_len, int32_t n)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t * segment_loss, int32_t num_segment_id1, int32_t num_segment_id2)}
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* segment_ids_mask, int32_t m, int32_t n)}

/* best_path result retrieval functions */
%feature("autodoc", "best_path_get_scores(self) -> numpy 1dim array of float") best_path_get_scores;
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** scores, int32_t* n)};
%feature("autodoc", "best_path_get_states(self) -> numpy 2dim array of int") best_path_get_states;
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** states, int32_t* m, int32_t* n)};
%feature("autodoc", "best_path_get_positions(self) -> numpy 2dim array of int") best_path_get_positions;
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** positions, int32_t* m, int32_t* n)};

/* best_path_trans_deriv result retrieval functions */
%feature("autodoc", "best_path_get_losses(self) -> numpy 1dim array of float") best_path_get_losses;
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** my_losses, int32_t* seq_len)}
#endif

%include "structure/DynProg.h"
