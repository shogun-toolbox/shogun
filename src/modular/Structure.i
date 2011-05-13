/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
%define DOCSTR
"The `Structure` module gathers all structure related learners available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Structure

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Structure_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
 #include <shogun/structure/PlifBase.h>
 #include <shogun/structure/Plif.h>
 #include <shogun/structure/PlifArray.h>
 #include <shogun/structure/DynProg.h>
 #include <shogun/structure/PlifMatrix.h>
 #include <shogun/structure/IntronList.h>
 #include <shogun/structure/SegmentLoss.h>
%}

/* Typemaps */
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_limits, int32_t p_len)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_penalties, int32_t p_len)};

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

/* PlifMatrix */
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* state_signals, int32_t feat_dim3, int32_t num_states)};


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


%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* seg_path, int32_t rows, int32_t cols)};


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

%apply (float64_t* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS) {(float64_t* seq, int32_t* dims, int32_t ndims)}
%apply (double* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS) {(double* seq, int32_t* dims, int32_t ndims)}

%apply (float64_t* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS) {(float64_t* penalties_array, int32_t* Dim, int32_t numDims)}
%apply (double* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS) {(double* penalties_array, int32_t* Dim, int32_t numDims)}


/* plif matrix functions */
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* ids, int32_t num_ids)}
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* min_values, int32_t num_values)}
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* max_values, int32_t num_values)}
%apply (bool* IN_ARRAY1, int32_t DIM1) {(bool* use_cache, int32_t num_values)}
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* use_svm, int32_t num_values)}
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* limits, int32_t num_plifs, int32_t num_limits)}
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* penalties, int32_t num_plifs, int32_t num_limits)}
%apply (SGString<char>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(SGString<char>* names, int32_t num_values, int32_t maxlen)}
%apply (SGString<char>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(SGString<char>* transform_type, int32_t num_values, int32_t maxlen)}

/* Remove C Prefix */
%rename(PlifBase) CPlifBase;
%rename(Plif) CPlif;
%rename(PlifArray) CPlifArray;
%rename(DynProg) CDynProg;
%rename(PlifMatrix) CPlifMatrix;
%rename(SegmentLoss) CSegmentLoss;
%rename(IntronList) CIntronList;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/structure/PlifBase.h>
%include <shogun/structure/Plif.h>
%include <shogun/structure/PlifArray.h>
%include <shogun/structure/DynProg.h>
%include <shogun/structure/PlifMatrix.h>
%include <shogun/structure/IntronList.h>
%include <shogun/structure/SegmentLoss.h>
