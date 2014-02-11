/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifdef HAVE_PYTHON
%feature("autodoc", "get_kernel_matrix(self) -> numpy 2dim array of float") get_kernel_matrix;
%feature("autodoc", "get_POIM2(self) -> [] of float") get_POIM2;
#endif

%ignore CWeightedDegreePositionStringKernel::set_position_weights(float64_t*);

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorKernel;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

#ifndef SWIGPYTHON
#define PROTOCOLS_CUSTOMKERNEL(class_name, type_name, format_str, typecode)
#define EXTEND_CUSTOMKERNEL(class_name, type_name, typecode)
#endif

/* Remove C Prefix */
%rename(Kernel) CKernel;
%rename(KernelNormalizer) CKernelNormalizer;
%rename(PyramidChi2) CPyramidChi2;
%rename(ANOVAKernel) CANOVAKernel;
%rename(AUCKernel) CAUCKernel;
%rename(AvgDiagKernelNormalizer) CAvgDiagKernelNormalizer;
%rename(RidgeKernelNormalizer) CRidgeKernelNormalizer;
%rename(CircularKernel) CCircularKernel;
%rename(Chi2Kernel) CChi2Kernel;
%rename(CombinedKernel) CCombinedKernel;
%rename(ProductKernel) CProductKernel;
%rename(CommUlongStringKernel) CCommUlongStringKernel;
%rename(CommWordStringKernel) CCommWordStringKernel;
%rename(ConstKernel) CConstKernel;

PROTOCOLS_CUSTOMKERNEL(CustomKernel, float32_t, "f\0", NPY_FLOAT32)
%rename(CustomKernel) CCustomKernel;

%rename(DiagKernel) CDiagKernel;
%rename(DistantSegmentsKernel) CDistantSegmentsKernel;
%rename(WaveKernel) CWaveKernel;
%rename(CauchyKernel) CCauchyKernel;
%rename(DiceKernelNormalizer) CDiceKernelNormalizer;
%rename(ExponentialKernel) CExponentialKernel;
%rename(ScatterKernelNormalizer) CScatterKernelNormalizer;
%rename(VarianceKernelNormalizer) CVarianceKernelNormalizer;
%rename(DistanceKernel) CDistanceKernel;
%rename(FixedDegreeStringKernel) CFixedDegreeStringKernel;
%rename(GaussianKernel) CGaussianKernel;
%rename(DirectorKernel) CDirectorKernel;
%rename(WaveletKernel) CWaveletKernel;
%rename(GaussianShiftKernel) CGaussianShiftKernel;
%rename(GaussianShortRealKernel) CGaussianShortRealKernel;
%rename(HistogramIntersectionKernel) CHistogramIntersectionKernel;
%rename(HistogramWordStringKernel) CHistogramWordStringKernel;
%rename(IdentityKernelNormalizer) CIdentityKernelNormalizer;
%rename(InverseMultiQuadricKernel) CInverseMultiQuadricKernel;
%rename(LinearKernel) CLinearKernel;
%rename(LinearStringKernel) CLinearStringKernel;
%rename(SparseSpatialSampleStringKernel) CSparseSpatialSampleStringKernel;
%rename(SplineKernel) CSplineKernel;
%rename(LocalAlignmentStringKernel) CLocalAlignmentStringKernel;
%rename(LocalityImprovedStringKernel) CLocalityImprovedStringKernel;
%rename(MatchWordStringKernel) CMatchWordStringKernel;
%rename(OligoStringKernel) COligoStringKernel;
%rename(PolyKernel) CPolyKernel;
%rename(PolyMatchStringKernel) CPolyMatchStringKernel;
%rename(PowerKernel) CPowerKernel;
%rename(LogKernel) CLogKernel;
%rename(GaussianMatchStringKernel) CGaussianMatchStringKernel;
%rename(SNPStringKernel) CSNPStringKernel;
%rename(RegulatoryModulesStringKernel) CRegulatoryModulesStringKernel;
%rename(PolyMatchWordStringKernel) CPolyMatchWordStringKernel;
%rename(SalzbergWordStringKernel) CSalzbergWordStringKernel;
%rename(SigmoidKernel) CSigmoidKernel;
%rename(SphericalKernel) CSphericalKernel;
%rename(SimpleLocalityImprovedStringKernel) CSimpleLocalityImprovedStringKernel;
%rename(SqrtDiagKernelNormalizer) CSqrtDiagKernelNormalizer;
%rename(TanimotoKernelNormalizer) CTanimotoKernelNormalizer;
%rename(TensorProductPairKernel) CTensorProductPairKernel;
%rename(TStudentKernel) CTStudentKernel;
%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;
%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;
%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;
%rename(WeightedDegreeRBFKernel) CWeightedDegreeRBFKernel;
%rename(SpectrumMismatchRBFKernel) CSpectrumMismatchRBFKernel;
%rename(ZeroMeanCenterKernelNormalizer) CZeroMeanCenterKernelNormalizer;
%rename(DotKernel) CDotKernel;
%rename(RationalQuadraticKernel) CRationalQuadraticKernel;
%rename(MultiquadricKernel) CMultiquadricKernel;
%rename(JensenShannonKernel) CJensenShannonKernel;
%rename(LinearARDKernel) CLinearARDKernel;
%rename(GaussianARDKernel) CGaussianARDKernel;
%rename(StringSubsequenceKernel) CStringSubsequenceKernel;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/kernel/Kernel.h>

%include <shogun/kernel/DotKernel.h>

/* Templated Class SparseKernel */
%include <shogun/kernel/SparseKernel.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(SparseRealKernel) CSparseKernel<float64_t>;
#endif
#ifdef USE_UINT16
    %template(SparseWordKernel) CSparseKernel<uint16_t>;
#endif
}

/* Templated Class StringKernel */
%include <shogun/kernel/string/StringKernel.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(StringRealKernel) CStringKernel<float64_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordKernel) CStringKernel<uint16_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharKernel) CStringKernel<char>;
#endif
#ifdef USE_UINT32
    %template(StringIntKernel) CStringKernel<int32_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongKernel) CStringKernel<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(StringShortKernel) CStringKernel<int16_t>;
#endif
#ifdef USE_UINT8
    %template(StringByteKernel) CStringKernel<uint8_t>;
#endif
}

%include <shogun/kernel/normalizer/KernelNormalizer.h>
%include <shogun/kernel/normalizer/AvgDiagKernelNormalizer.h>
%include <shogun/kernel/normalizer/RidgeKernelNormalizer.h>
%include <shogun/kernel/normalizer/DiceKernelNormalizer.h>
%include <shogun/kernel/normalizer/ScatterKernelNormalizer.h>
%include <shogun/kernel/normalizer/VarianceKernelNormalizer.h>
%include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
%include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
%include <shogun/kernel/normalizer/TanimotoKernelNormalizer.h>
%include <shogun/kernel/normalizer/ZeroMeanCenterKernelNormalizer.h>

%include <shogun/kernel/PyramidChi2.h>
%include <shogun/kernel/ANOVAKernel.h>
%include <shogun/kernel/AUCKernel.h>
%include <shogun/kernel/CauchyKernel.h>
%include <shogun/kernel/CircularKernel.h>
%include <shogun/kernel/Chi2Kernel.h>
%include <shogun/kernel/CombinedKernel.h>
%include <shogun/kernel/ProductKernel.h>
%include <shogun/kernel/string/CommUlongStringKernel.h>
%include <shogun/kernel/string/CommWordStringKernel.h>
%include <shogun/kernel/ConstKernel.h>
%include <shogun/kernel/CustomKernel.h>
%include <shogun/kernel/DiagKernel.h>
%include <shogun/kernel/DistanceKernel.h>
%include <shogun/kernel/string/DistantSegmentsKernel.h>
%include <shogun/kernel/ExponentialKernel.h>
%include <shogun/kernel/string/FixedDegreeStringKernel.h>
%include <shogun/kernel/GaussianKernel.h>
%include <shogun/kernel/DirectorKernel.h>
%include <shogun/kernel/GaussianShiftKernel.h>
%include <shogun/kernel/GaussianShortRealKernel.h>
%include <shogun/kernel/HistogramIntersectionKernel.h>
%include <shogun/kernel/string/HistogramWordStringKernel.h>
%include <shogun/kernel/InverseMultiQuadricKernel.h>
%include <shogun/kernel/LinearKernel.h>
%include <shogun/kernel/string/LinearStringKernel.h>
%include <shogun/kernel/string/SparseSpatialSampleStringKernel.h>
%include <shogun/kernel/string/LocalAlignmentStringKernel.h>
%include <shogun/kernel/string/LocalityImprovedStringKernel.h>
%include <shogun/kernel/string/MatchWordStringKernel.h>
%include <shogun/kernel/string/OligoStringKernel.h>
%include <shogun/kernel/PolyKernel.h>
%include <shogun/kernel/string/PolyMatchStringKernel.h>
%include <shogun/kernel/PowerKernel.h>
%include <shogun/kernel/LogKernel.h>
%include <shogun/kernel/string/GaussianMatchStringKernel.h>
%include <shogun/kernel/string/SNPStringKernel.h>
%include <shogun/kernel/string/RegulatoryModulesStringKernel.h>
%include <shogun/kernel/string/PolyMatchWordStringKernel.h>
%include <shogun/kernel/string/SalzbergWordStringKernel.h>
%include <shogun/kernel/SigmoidKernel.h>
%include <shogun/kernel/string/SimpleLocalityImprovedStringKernel.h>
%include <shogun/kernel/SphericalKernel.h>
%include <shogun/kernel/SplineKernel.h>
%include <shogun/kernel/TensorProductPairKernel.h>
%include <shogun/kernel/TStudentKernel.h>
%include <shogun/kernel/WaveKernel.h>
%include <shogun/kernel/WaveletKernel.h>
%include <shogun/kernel/string/WeightedCommWordStringKernel.h>
%include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
%include <shogun/kernel/string/WeightedDegreeStringKernel.h>
%include <shogun/kernel/WeightedDegreeRBFKernel.h>
%include <shogun/kernel/string/SpectrumMismatchRBFKernel.h>
%include <shogun/kernel/MultiquadricKernel.h>
%include <shogun/kernel/RationalQuadraticKernel.h>
%include <shogun/kernel/JensenShannonKernel.h>
%include <shogun/kernel/LinearARDKernel.h>
%include <shogun/kernel/GaussianARDKernel.h>
%include <shogun/kernel/string/StringSubsequenceKernel.h>

EXTEND_CUSTOMKERNEL(CustomKernel, float32_t, NPY_FLOAT32)
