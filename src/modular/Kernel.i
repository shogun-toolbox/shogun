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
"The `Kernel` module gathers all kernels available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Kernel

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Kernel_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_kernel_matrix(self) -> numpy 2dim array of float") get_kernel_matrix;
%feature("autodoc", "get_POIM2(self) -> [] of float") get_POIM2;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/KernelNormalizer.h>
#include <shogun/kernel/PyramidChi2.h>
#include <shogun/kernel/AUCKernel.h>
#include <shogun/kernel/AvgDiagKernelNormalizer.h>
#include <shogun/kernel/Chi2Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CommUlongStringKernel.h>
#include <shogun/kernel/CommWordStringKernel.h>
#include <shogun/kernel/ConstKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/DiceKernelNormalizer.h>
#include <shogun/kernel/DistanceKernel.h>
#include <shogun/kernel/FixedDegreeStringKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/GaussianShiftKernel.h>
#include <shogun/kernel/GaussianShortRealKernel.h>
#include <shogun/kernel/HistogramWordStringKernel.h>
#include <shogun/kernel/IdentityKernelNormalizer.h>
#include <shogun/kernel/SimpleKernel.h>
#include <shogun/kernel/LinearByteKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/LinearStringKernel.h>
#include <shogun/kernel/LinearWordKernel.h>
#include <shogun/kernel/LocalAlignmentStringKernel.h>
#include <shogun/kernel/LocalityImprovedStringKernel.h>
#include <shogun/kernel/MatchWordStringKernel.h> 
#include <shogun/kernel/MindyGramKernel.h>
#include <shogun/kernel/OligoKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/PolyMatchStringKernel.h>
#include <shogun/kernel/PolyMatchWordStringKernel.h>
#include <shogun/kernel/SalzbergWordStringKernel.h>
#include <shogun/kernel/SigmoidKernel.h>
#include <shogun/kernel/SimpleLocalityImprovedStringKernel.h>
#include <shogun/kernel/SparseGaussianKernel.h>
#include <shogun/kernel/SparseKernel.h>
#include <shogun/kernel/SparseLinearKernel.h>
#include <shogun/kernel/SparsePolyKernel.h>
#include <shogun/kernel/SqrtDiagKernelNormalizer.h>
#include <shogun/kernel/StringKernel.h>
#include <shogun/kernel/TanimotoKernelNormalizer.h>
#include <shogun/kernel/TensorProductPairKernel.h>
#include <shogun/kernel/WeightedCommWordStringKernel.h>
#include <shogun/kernel/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/WeightedDegreeStringKernel.h>
%}

/* Typemaps */
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* m, int32_t* n)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(const float64_t* km, int32_t rows, int32_t cols)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(const float64_t* km, int32_t len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst_w, int32_t* dst_dims)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src_w, int32_t src_w_dim)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* w, int32_t d)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* weights, int32_t d, int32_t len)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* shifts, int32_t len)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* pws, int32_t len)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* distrib, int32_t num_sym, int32_t num_feat)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** poim, int32_t* result_len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** weights, int32_t* num_weights)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_weights, int32_t d)};

%ignore CWeightedDegreePositionStringKernel::set_position_weights(float64_t*);

/* Remove C Prefix */
%rename(Kernel) CKernel;
%rename(KernelNormalizer) CKernelNormalizer;
%rename(PyramidChi2) CPyramidChi2;
%rename(AUCKernel) CAUCKernel;
%rename(AvgDiagKernelNormalizer) CAvgDiagKernelNormalizer;
%rename(Chi2Kernel) CChi2Kernel;
%rename(CombinedKernel) CCombinedKernel;
%rename(CommUlongStringKernel) CCommUlongStringKernel;
%rename(CommWordStringKernel) CCommWordStringKernel;
%rename(ConstKernel) CConstKernel;
%rename(CustomKernel) CCustomKernel;
%rename(DiagKernel) CDiagKernel;
%rename(DiceKernelNormalizer) CDiceKernelNormalizer;
%rename(DistanceKernel) CDistanceKernel;
%rename(FixedDegreeStringKernel) CFixedDegreeStringKernel;
%rename(GaussianKernel) CGaussianKernel;
%rename(GaussianShiftKernel) CGaussianShiftKernel;
%rename(GaussianShortRealKernel) CGaussianShortRealKernel;
%rename(HistogramWordStringKernel) CHistogramWordStringKernel;
%rename(IdentityKernelNormalizer) CIdentityKernelNormalizer;
%rename(LinearByteKernel) CLinearByteKernel;
%rename(LinearKernel) CLinearKernel;
%rename(LinearStringKernel) CLinearStringKernel;
%rename(LinearWordKernel) CLinearWordKernel;
%rename(LocalAlignmentStringKernel) CLocalAlignmentStringKernel;
%rename(LocalityImprovedStringKernel) CLocalityImprovedStringKernel;
%rename(MatchWordStringKernel) CMatchWordStringKernel;
#ifdef HAVE_MINDY
%rename (MindyGramKernel) CMindyGramKernel;
#endif
%rename(OligoKernel) COligoKernel;
%rename(PolyKernel) CPolyKernel;
%rename(PolyMatchStringKernel) CPolyMatchStringKernel;
%rename(PolyMatchWordStringKernel) CPolyMatchWordStringKernel;
%rename(SalzbergWordStringKernel) CSalzbergWordStringKernel;
%rename(SigmoidKernel) CSigmoidKernel;
%rename(SimpleLocalityImprovedStringKernel) CSimpleLocalityImprovedStringKernel;
%rename(SparseGaussianKernel) CSparseGaussianKernel;
%rename(SparseLinearKernel) CSparseLinearKernel;
%rename(SparsePolyKernel) CSparsePolyKernel;
%rename(SqrtDiagKernelNormalizer) CSqrtDiagKernelNormalizer;
%rename(TanimotoKernelNormalizer) CTanimotoKernelNormalizer;
%rename(TensorProductPairKernel) CTensorProductPairKernel;
%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;
%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;
%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/kernel/Kernel.h>

/* Templated Class SimpleKernel */
%include <shogun/kernel/SimpleKernel.h>
%template(RealKernel) CSimpleKernel<float64_t>;
%template(ShortRealKernel) CSimpleKernel<float32_t>;
%template(WordKernel) CSimpleKernel<uint16_t>;
%template(CharKernel) CSimpleKernel<char>;
%template(ByteKernel) CSimpleKernel<uint8_t>;
%template(IntKernel) CSimpleKernel<int32_t>;
%template(ShortKernel) CSimpleKernel<int16_t>;
%template(UlongKernel) CSimpleKernel<uint64_t>;

/* Templated Class SparseKernel */
%include <shogun/kernel/SparseKernel.h>
%template(SparseRealKernel) CSparseKernel<float64_t>;
%template(SparseWordKernel) CSparseKernel<uint16_t>;

/* Templated Class StringKernel */
%include <shogun/kernel/StringKernel.h>
%template(StringRealKernel) CStringKernel<float64_t>;
%template(StringWordKernel) CStringKernel<uint16_t>;
%template(StringCharKernel) CStringKernel<char>;
%template(StringIntKernel) CStringKernel<int32_t>;
%template(StringUlongKernel) CStringKernel<uint64_t>;
%template(StringShortKernel) CStringKernel<int16_t>;
%template(StringByteKernel) CStringKernel<uint8_t>;

%include <shogun/kernel/KernelNormalizer.h>
%include <shogun/kernel/PyramidChi2.h>
%include <shogun/kernel/AUCKernel.h> 
%include <shogun/kernel/AvgDiagKernelNormalizer.h>
%include <shogun/kernel/Chi2Kernel.h>
%include <shogun/kernel/CombinedKernel.h>
%include <shogun/kernel/CommUlongStringKernel.h>
%include <shogun/kernel/CommWordStringKernel.h>
%include <shogun/kernel/ConstKernel.h>
%include <shogun/kernel/CustomKernel.h>
%include <shogun/kernel/DiagKernel.h>
%include <shogun/kernel/DiceKernelNormalizer.h>
%include <shogun/kernel/DistanceKernel.h>
%include <shogun/kernel/FixedDegreeStringKernel.h>
%include <shogun/kernel/GaussianKernel.h>
%include <shogun/kernel/GaussianShiftKernel.h>
%include <shogun/kernel/GaussianShortRealKernel.h>
%include <shogun/kernel/HistogramWordStringKernel.h>
%include <shogun/kernel/IdentityKernelNormalizer.h>
%include <shogun/kernel/LinearByteKernel.h>
%include <shogun/kernel/LinearKernel.h>
%include <shogun/kernel/LinearStringKernel.h>
%include <shogun/kernel/LinearWordKernel.h>
%include <shogun/kernel/LocalAlignmentStringKernel.h>
%include <shogun/kernel/LocalityImprovedStringKernel.h>
%include <shogun/kernel/MatchWordStringKernel.h>
%include <shogun/kernel/MindyGramKernel.h>
%include <shogun/kernel/OligoKernel.h>
%include <shogun/kernel/PolyKernel.h>
%include <shogun/kernel/PolyMatchStringKernel.h>
%include <shogun/kernel/PolyMatchWordStringKernel.h>
%include <shogun/kernel/SalzbergWordStringKernel.h>
%include <shogun/kernel/SigmoidKernel.h>
%include <shogun/kernel/SimpleLocalityImprovedStringKernel.h>
%include <shogun/kernel/SparseGaussianKernel.h>
%include <shogun/kernel/SparseLinearKernel.h>
%include <shogun/kernel/SparsePolyKernel.h>
%include <shogun/kernel/SqrtDiagKernelNormalizer.h>
%include <shogun/kernel/TanimotoKernelNormalizer.h>
%include <shogun/kernel/TensorProductPairKernel.h>
%include <shogun/kernel/WeightedCommWordStringKernel.h>
%include <shogun/kernel/WeightedDegreePositionStringKernel.h>
%include <shogun/kernel/WeightedDegreeStringKernel.h>
