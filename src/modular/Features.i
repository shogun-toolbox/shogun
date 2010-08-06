/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */


/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Features_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/StringFileFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/features/AttributeFeatures.h>
#include <shogun/features/Alphabet.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/features/Labels.h>
#include <shogun/features/RealFileFeatures.h>
#include <shogun/features/FKFeatures.h>
#include <shogun/features/TOPFeatures.h>
#include <shogun/features/SNPFeatures.h>
#include <shogun/features/WDFeatures.h>
#include <shogun/features/HashedWDFeatures.h>
#include <shogun/features/HashedWDFeaturesTransposed.h>
#include <shogun/features/PolyFeatures.h>
#include <shogun/features/SparsePolyFeatures.h>
#include <shogun/features/LBPPyrDotFeatures.h>
#include <shogun/features/ExplicitSpecFeatures.h>
#include <shogun/features/ImplicitWeightedSpecFeatures.h>
%}

/* These functions return new Objects */
%newobject get_transposed();

/* Typemaps */

%apply (bool* IN_ARRAY1, int32_t DIM1) {(bool* src, int32_t len)};
%apply (char* IN_ARRAY1, int32_t DIM1) {(char* src, int32_t len)};
%apply (uint8_t* IN_ARRAY1, int32_t DIM1) {(uint8_t* src, int32_t len)};
%apply (int16_t* IN_ARRAY1, int32_t DIM1) {(int16_t* src, int32_t len)};
%apply (uint16_t* IN_ARRAY1, int32_t DIM1) {(uint16_t* src, int32_t len)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* src, int32_t len)};
%apply (uint32_t* IN_ARRAY1, int32_t DIM1) {(uint32_t* src, int32_t len)};
%apply (int64_t* IN_ARRAY1, int32_t DIM1) {(int64_t* src, int32_t len)};
%apply (uint64_t* IN_ARRAY1, int32_t DIM1) {(uint64_t* src, int32_t len)};
%apply (float32_t* IN_ARRAY1, int32_t DIM1) {(float32_t* src, int32_t len)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src, int32_t len)};
%apply (floatmax_t* IN_ARRAY1, int32_t DIM1) {(floatmax_t* src, int32_t len)};

%apply (bool** ARGOUT1, int32_t* DIM1) {(bool** dst, int32_t* len)};
%apply (char** ARGOUT1, int32_t* DIM1) {(char** dst, int32_t* len)};
%apply (uint8_t** ARGOUT1, int32_t* DIM1) {(uint8_t** dst, int32_t* len)};
%apply (int16_t** ARGOUT1, int32_t* DIM1) {(int16_t** dst, int32_t* len)};
%apply (uint16_t** ARGOUT1, int32_t* DIM1) {(uint16_t** dst, int32_t* len)};
%apply (int32_t** ARGOUT1, int32_t* DIM1) {(int32_t** dst, int32_t* len)};
%apply (uint32_t** ARGOUT1, int32_t* DIM1) {(uint32_t** dst, int32_t* len)};
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(int64_t** dst, int32_t* len)};
%apply (uint64_t** ARGOUT1, int32_t* DIM1) {(uint64_t** dst, int32_t* len)};
%apply (float32_t** ARGOUT1, int32_t* DIM1) {(float32_t** dst, int32_t* len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* len)};
%apply (floatmax_t** ARGOUT1, int32_t* DIM1) {(floatmax_t** dst, int32_t* len)};

%apply (bool* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(bool* src, int32_t num_feat, int32_t num_vec)};
%apply (char* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(char* src, int32_t num_feat, int32_t num_vec)};
%apply (uint8_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint8_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int16_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint16_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int64_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint64_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* src, int32_t num_feat, int32_t num_vec)};
%apply (floatmax_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(floatmax_t* src, int32_t num_feat, int32_t num_vec)};

%apply (bool** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(bool** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (char** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(char** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint8_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint8_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (int16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int16_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint16_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint32_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (int64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int64_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint64_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (float32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float32_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (floatmax_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(floatmax_t** dst, int32_t* num_feat, int32_t* num_vec)};

%apply (shogun::T_STRING<bool>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<bool>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<char>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<char>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint8_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<uint8_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<int16_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<int16_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint16_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<uint16_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<int32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<int32_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<uint32_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<int64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<int64_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<uint64_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<float32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<float32_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<float64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<float64_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<floatmax_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<floatmax_t>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};

%apply (shogun::T_STRING<bool>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<bool>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<char>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<char>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint8_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<uint8_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<int16_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<int16_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint16_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<uint16_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<int32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<int32_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<uint32_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<int64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<int64_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<uint64_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<float32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<float32_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<float64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<float64_t>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<floatmax_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<floatmax_t>** dst, int32_t* num_str)};

%apply (shogun::TSparse<bool>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<bool>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<char>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<char>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint8_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<uint8_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<int16_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<int16_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint16_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<uint16_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<int32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<int32_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<uint32_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<int64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<int64_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<uint64_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<float32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<float32_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<float64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<float64_t>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<floatmax_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<floatmax_t>* src, int32_t num_feat, int32_t num_vec)};

%apply (shogun::TSparse<bool>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<bool>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<char>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<char>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint8_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<uint8_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<int16_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<int16_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint16_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<uint16_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<int32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<int32_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<uint32_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<int64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<int64_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<uint64_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<float32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<float32_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<float64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<float64_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<floatmax_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<floatmax_t>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* weights, int32_t d)};

/* There seems to be a bug(?) in swig type reduction, e.g. int32_t** does not
 * match int**, therefore apply's are repeated here for ordinary types. While
 * this might introduce compile failures when types mismatch, it should never
 * generate invalid code. */
%apply (uint8_t* IN_ARRAY1, int32_t DIM1) {(unsigned char* src, int32_t len)};
%apply (int16_t* IN_ARRAY1, int32_t DIM1) {(short int* src, int32_t len)};
%apply (uint16_t* IN_ARRAY1, int32_t DIM1) {(unsigned short int* src, int32_t len)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int* src, int32_t len)};
%apply (uint32_t* IN_ARRAY1, int32_t DIM1) {(unsigned int* src, int32_t len)};
#ifdef SWIGWORDSIZE64
%apply (int64_t* IN_ARRAY1, int32_t DIM1) {(long int* src, int32_t len)};
%apply (uint64_t* IN_ARRAY1, int32_t DIM1) {(unsigned long int* src, int32_t len)};
#else
%apply (int64_t* IN_ARRAY1, int32_t DIM1) {(long long int* src, int32_t len)};
%apply (uint64_t* IN_ARRAY1, int32_t DIM1) {(unsigned long long int* src, int32_t len)};
#endif
%apply (float32_t* IN_ARRAY1, int32_t DIM1) {(float* src, int32_t len)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(double* src, int32_t len)};
%apply (floatmax_t* IN_ARRAY1, int32_t DIM1) {(long double* src, int32_t len)};

%apply (uint8_t** ARGOUT1, int32_t* DIM1) {(unsigned char** dst, int32_t* len)};
%apply (int16_t** ARGOUT1, int32_t* DIM1) {(short int** dst, int32_t* len)};
%apply (uint16_t** ARGOUT1, int32_t* DIM1) {(unsigned short int** dst, int32_t* len)};
%apply (int32_t** ARGOUT1, int32_t* DIM1) {(int** dst, int32_t* len)};
%apply (uint32_t** ARGOUT1, int32_t* DIM1) {(unsigned int** dst, int32_t* len)};
#ifdef SWIGWORDSIZE64
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(long int** dst, int32_t* len)};
%apply (uint64_t** ARGOUT1, int32_t* DIM1) {(unsigned long int** dst, int32_t* len)};
#else
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(long long int** dst, int32_t* len)};
%apply (uint64_t** ARGOUT1, int32_t* DIM1) {(unsigned long long int** dst, int32_t* len)};
#endif
%apply (float32_t** ARGOUT1, int32_t* DIM1) {(float** dst, int32_t* len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(double** dst, int32_t* len)};
%apply (floatmax_t** ARGOUT1, int32_t* DIM1) {(long double** dst, int32_t* len)};

%apply (uint8_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(unsigned char* src, int32_t num_feat, int32_t num_vec)};
%apply (int16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(short int* src, int32_t num_feat, int32_t num_vec)};
%apply (uint16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(unsigned short int* src, int32_t num_feat, int32_t num_vec)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int* src, int32_t num_feat, int32_t num_vec)};
%apply (uint32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(unsigned int* src, int32_t num_feat, int32_t num_vec)};
#ifdef SWIGWORDSIZE64
%apply (int64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(long int* src, int32_t num_feat, int32_t num_vec)};
%apply (uint64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(unsigned long int* src, int32_t num_feat, int32_t num_vec)};
#else
%apply (int64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(long long int* src, int32_t num_feat, int32_t num_vec)};
%apply (uint64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(unsigned long long int* src, int32_t num_feat, int32_t num_vec)};
#endif
%apply (float32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float* src, int32_t num_feat, int32_t num_vec)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(double* src, int32_t num_feat, int32_t num_vec)};

%apply (uint8_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(unsigned char** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (int16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(short int** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(unsigned short int** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(unsigned int** dst, int32_t* num_feat, int32_t* num_vec)};
#ifdef SWIGWORDSIZE64
%apply (int64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(long int** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(unsigned long int** dst, int32_t* num_feat, int32_t* num_vec)};
#else
%apply (int64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(long long int** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (uint64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(unsigned long long int** dst, int32_t* num_feat, int32_t* num_vec)};
#endif
%apply (float32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(double** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (floatmax_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(long double** dst, int32_t* num_feat, int32_t* num_vec)};

%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* hist, int32_t rows, int32_t cols)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** hist, int32_t* rows, int32_t* cols)};
%apply (shogun::T_STRING<uint8_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<unsigned char>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<int16_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<short int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint16_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<unsigned short int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<int32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<unsigned int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
#ifdef SWIGWORDSIZE64
%apply (shogun::T_STRING<int64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<long int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<unsigned long int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
#else
%apply (shogun::T_STRING<int64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<long long int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<uint64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<unsigned long long int>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
#endif
%apply (shogun::T_STRING<float32_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<float>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<float64_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<double>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};
%apply (shogun::T_STRING<floatmax_t>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {(shogun::T_STRING<long double>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)};

%apply (shogun::T_STRING<uint8_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<unsigned char>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<int16_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<short int>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint16_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<unsigned short int>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<int32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<int>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<unsigned int>** dst, int32_t* num_str)};
#ifdef SWIGWORDSIZE64
%apply (shogun::T_STRING<int64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<long int>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<unsigned long int>** dst, int32_t* num_str)};
#else
%apply (shogun::T_STRING<int64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<long long int>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<uint64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<unsigned long long int>** dst, int32_t* num_str)};
#endif
%apply (shogun::T_STRING<float32_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<float>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<float64_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<double>** dst, int32_t* num_str)};
%apply (shogun::T_STRING<floatmax_t>** ARGOUT_STRINGS, int32_t* NUM) {(shogun::T_STRING<long double>** dst, int32_t* num_str)};

%apply (shogun::TSparse<uint8_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<unsigned char>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<int16_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<short int>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint16_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<unsigned short int>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<int32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<int>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<unsigned int>* src, int32_t num_feat, int32_t num_vec)};
#ifdef SWIGWORDSIZE64
%apply (shogun::TSparse<int64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<long int>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<unsigned long int>* src, int32_t num_feat, int32_t num_vec)};
#else
%apply (shogun::TSparse<int64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<long long int>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<uint64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<unsigned long long int>* src, int32_t num_feat, int32_t num_vec)};
#endif
%apply (shogun::TSparse<float32_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<float>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<float64_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<double>* src, int32_t num_feat, int32_t num_vec)};
%apply (shogun::TSparse<floatmax_t>* IN_SPARSE, int32_t DIM1, int32_t DIM2) {(shogun::TSparse<long double>* src, int32_t num_feat, int32_t num_vec)};

%apply (shogun::TSparse<uint8_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<unsigned char>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<int16_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<short int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint16_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<unsigned short int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<int32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<unsigned int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
#ifdef SWIGWORDSIZE64
%apply (shogun::TSparse<int64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<long int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<unsigned long int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
#else
%apply (shogun::TSparse<int64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<long long int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<uint64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<unsigned long long int>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
#endif
%apply (shogun::TSparse<float32_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<float>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<float64_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<double>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};
%apply (shogun::TSparse<floatmax_t>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {(shogun::TSparse<long double>** dst, int32_t* num_feat, int32_t* num_vec, int64_t* nnz)};

/* label confidences */
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* out_num_labels, int32_t* out_num_classes)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(double** dst, int32_t* num_feat, int32_t* num_vec)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* in_confidences, int32_t in_num_labels, int32_t in_num_classes)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(double* in_confidences, int32_t in_num_labels, int32_t in_num_classes)};

/* Remove C Prefix */
%rename(Features) CFeatures;
%rename(DotFeatures) CDotFeatures;
%rename(DummyFeatures) CDummyFeatures;
%rename(AttributeFeatures) CAttributeFeatures;
%rename(Alphabet) CAlphabet;
%rename(CombinedFeatures) CCombinedFeatures;
%rename(CombinedDotFeatures) CCombinedDotFeatures;
%rename(Labels) CLabels;
%rename(RealFileFeatures) CRealFileFeatures;
%rename(FKFeatures) CFKFeatures;
%rename(TOPFeatures) CTOPFeatures;
%rename(SNPFeatures) CSNPFeatures;
%rename(WDFeatures) CWDFeatures;
%rename(HashedWDFeatures) CHashedWDFeatures;
%rename(HashedWDFeaturesTransposed) CHashedWDFeaturesTransposed;
%rename(PolyFeatures) CPolyFeatures;
%rename(SparsePolyFeatures) CSparsePolyFeatures;
%rename(LBPPyrDotFeatures) CLBPPyrDotFeatures;
%rename(ExplicitSpecFeatures) CExplicitSpecFeatures;
%rename(ImplicitWeightedSpecFeatures) CImplicitWeightedSpecFeatures;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>

/* Templated Class StringFeatures */
%include <shogun/features/StringFeatures.h>
namespace shogun
{
    %template(StringBoolFeatures) CStringFeatures<bool>;
    %template(StringCharFeatures) CStringFeatures<char>;
    %template(StringByteFeatures) CStringFeatures<uint8_t>;
    %template(StringShortFeatures) CStringFeatures<int16_t>;
    %template(StringWordFeatures) CStringFeatures<uint16_t>;
    %template(StringIntFeatures) CStringFeatures<int32_t>;
    %template(StringUIntFeatures) CStringFeatures<uint32_t>;
    %template(StringLongFeatures) CStringFeatures<int64_t>;
    %template(StringUlongFeatures) CStringFeatures<uint64_t>;
    %template(StringShortRealFeatures) CStringFeatures<float32_t>;
    %template(StringRealFeatures) CStringFeatures<float64_t>;
    %template(StringLongRealFeatures) CStringFeatures<floatmax_t>;
}

/* Templated Class StringFileFeatures */
%include <shogun/features/StringFileFeatures.h>
namespace shogun
{
    %template(StringFileBoolFeatures) CStringFileFeatures<bool>;
    %template(StringFileCharFeatures) CStringFileFeatures<char>;
    %template(StringFileByteFeatures) CStringFileFeatures<uint8_t>;
    %template(StringFileShortFeatures) CStringFileFeatures<int16_t>;
    %template(StringFileWordFeatures) CStringFileFeatures<uint16_t>;
    %template(StringFileIntFeatures) CStringFileFeatures<int32_t>;
    %template(StringFileUIntFeatures) CStringFileFeatures<uint32_t>;
    %template(StringFileLongFeatures) CStringFileFeatures<int64_t>;
    %template(StringFileUlongFeatures) CStringFileFeatures<uint64_t>;
    %template(StringFileShortRealFeatures) CStringFileFeatures<float32_t>;
    %template(StringFileRealFeatures) CStringFileFeatures<float64_t>;
    %template(StringFileLongRealFeatures) CStringFileFeatures<floatmax_t>;
}

/* Templated Class SparseFeatures */
%include <shogun/features/SparseFeatures.h>
namespace shogun
{
    %template(SparseBoolFeatures) CSparseFeatures<bool>;
    %template(SparseCharFeatures) CSparseFeatures<char>;
    %template(SparseByteFeatures) CSparseFeatures<uint8_t>;
    %template(SparseShortFeatures) CSparseFeatures<int16_t>;
    %template(SparseWordFeatures) CSparseFeatures<uint16_t>;
    %template(SparseIntFeatures) CSparseFeatures<int32_t>;
    %template(SparseUIntFeatures) CSparseFeatures<uint32_t>;
    %template(SparseLongFeatures) CSparseFeatures<int64_t>;
    %template(SparseUlongFeatures) CSparseFeatures<uint64_t>;
    %template(SparseShortRealFeatures) CSparseFeatures<float32_t>;
    %template(SparseRealFeatures) CSparseFeatures<float64_t>;
    %template(SparseLongRealFeatures) CSparseFeatures<floatmax_t>;
}

/* Templated Class SimpleFeatures */
%include <shogun/features/SimpleFeatures.h>
namespace shogun
{
    %template(BoolFeatures) CSimpleFeatures<bool>;
    %template(CharFeatures) CSimpleFeatures<char>;
    %template(ByteFeatures) CSimpleFeatures<uint8_t>;
    %template(WordFeatures) CSimpleFeatures<uint16_t>;
    %template(ShortFeatures) CSimpleFeatures<int16_t>;
    %template(IntFeatures)  CSimpleFeatures<int32_t>;
    %template(UIntFeatures)  CSimpleFeatures<uint32_t>;
    %template(LongIntFeatures)  CSimpleFeatures<int64_t>;
    %template(ULongIntFeatures)  CSimpleFeatures<uint64_t>;
    %template(ShortRealFeatures) CSimpleFeatures<float32_t>;
    %template(RealFeatures) CSimpleFeatures<float64_t>;
    %template(LongRealFeatures) CSimpleFeatures<floatmax_t>;
}

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/AttributeFeatures.h>
%include <shogun/features/Alphabet.h>
%include <shogun/features/CombinedFeatures.h>
%include <shogun/features/CombinedDotFeatures.h>

%include <shogun/features/Labels.h>
%include <shogun/features/RealFileFeatures.h>
%include <shogun/features/FKFeatures.h>
%include <shogun/features/TOPFeatures.h>
%include <shogun/features/SNPFeatures.h>
%include <shogun/features/WDFeatures.h>
%include <shogun/features/HashedWDFeatures.h>
%include <shogun/features/HashedWDFeaturesTransposed.h>
%include <shogun/features/PolyFeatures.h>
%include <shogun/features/SparsePolyFeatures.h>
%include <shogun/features/LBPPyrDotFeatures.h>
%include <shogun/features/ExplicitSpecFeatures.h>
%include <shogun/features/ImplicitWeightedSpecFeatures.h>
