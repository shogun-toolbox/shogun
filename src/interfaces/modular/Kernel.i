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
%rename(CommUlongStringKernel) CCommUlongStringKernel;
%rename(CommWordStringKernel) CCommWordStringKernel;
%rename(ConstKernel) CConstKernel;
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
%rename(MultitaskKernelNormalizer) CMultitaskKernelNormalizer;
%rename(MultitaskKernelMklNormalizer) CMultitaskKernelMklNormalizer;
%rename(MultitaskKernelTreeNormalizer) CMultitaskKernelTreeNormalizer;
%rename(MultitaskKernelMaskNormalizer) CMultitaskKernelMaskNormalizer;
%rename(MultitaskKernelMaskPairNormalizer) CMultitaskKernelMaskPairNormalizer;
%rename(MultitaskKernelPlifNormalizer) CMultitaskKernelPlifNormalizer;
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
%include <shogun/kernel/StringKernel.h>
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



/* Provide some cast functionality available to target languages */
%inline %{

/* C++-style cast */
shogun::CMultitaskKernelNormalizer *KernelNormalizerToMultitaskKernelNormalizer(shogun::CKernelNormalizer* n) {
   return dynamic_cast<shogun::CMultitaskKernelNormalizer*>(n);
}

shogun::CMultitaskKernelTreeNormalizer *KernelNormalizerToMultitaskKernelTreeNormalizer(shogun::CKernelNormalizer* n) {
   return dynamic_cast<shogun::CMultitaskKernelTreeNormalizer*>(n);
}

shogun::CMultitaskKernelMaskNormalizer *KernelNormalizerToMultitaskKernelMaskNormalizer(shogun::CKernelNormalizer* n) {
   return dynamic_cast<shogun::CMultitaskKernelMaskNormalizer*>(n);
}

shogun::CMultitaskKernelMaskPairNormalizer *KernelNormalizerToMultitaskKernelMaskPairNormalizer(shogun::CKernelNormalizer* n) {
   return dynamic_cast<shogun::CMultitaskKernelMaskPairNormalizer*>(n);
}

shogun::CMultitaskKernelPlifNormalizer *KernelNormalizerToMultitaskKernelPlifNormalizer(shogun::CKernelNormalizer* n) {
   return dynamic_cast<shogun::CMultitaskKernelPlifNormalizer*>(n);
}

shogun::CCombinedKernel *KernelToCombinedKernel(shogun::CKernel* n) {
   return dynamic_cast<shogun::CCombinedKernel*>(n);
}


%}

%include <shogun/kernel/KernelNormalizer.h>
%include <shogun/kernel/PyramidChi2.h>
%include <shogun/kernel/ANOVAKernel.h>
%include <shogun/kernel/AUCKernel.h>
%include <shogun/kernel/AvgDiagKernelNormalizer.h>
%include <shogun/kernel/RidgeKernelNormalizer.h>
%include <shogun/kernel/CauchyKernel.h>
%include <shogun/kernel/CircularKernel.h>
%include <shogun/kernel/Chi2Kernel.h>
%include <shogun/kernel/CombinedKernel.h>
%include <shogun/kernel/CommUlongStringKernel.h>
%include <shogun/kernel/CommWordStringKernel.h>
%include <shogun/kernel/ConstKernel.h>
%include <shogun/kernel/CustomKernel.h>
%include <shogun/kernel/DiagKernel.h>
%include <shogun/kernel/DiceKernelNormalizer.h>
%include <shogun/kernel/ScatterKernelNormalizer.h>
%include <shogun/kernel/VarianceKernelNormalizer.h>
%include <shogun/kernel/DistanceKernel.h>
%include <shogun/kernel/DistantSegmentsKernel.h>
%include <shogun/kernel/ExponentialKernel.h>
%include <shogun/kernel/FixedDegreeStringKernel.h>
%include <shogun/kernel/GaussianKernel.h>
%include <shogun/kernel/GaussianShiftKernel.h>
%include <shogun/kernel/GaussianShortRealKernel.h>
%include <shogun/kernel/HistogramIntersectionKernel.h>
%include <shogun/kernel/HistogramWordStringKernel.h>
%include <shogun/kernel/IdentityKernelNormalizer.h>
%include <shogun/kernel/InverseMultiQuadricKernel.h>
%include <shogun/kernel/LinearKernel.h>
%include <shogun/kernel/LinearStringKernel.h>
%include <shogun/kernel/SparseSpatialSampleStringKernel.h>
%include <shogun/kernel/LocalAlignmentStringKernel.h>
%include <shogun/kernel/LocalityImprovedStringKernel.h>
%include <shogun/kernel/MatchWordStringKernel.h>
%include <shogun/kernel/MultitaskKernelNormalizer.h>
%include <shogun/kernel/MultitaskKernelMklNormalizer.h>
%include <shogun/kernel/MultitaskKernelTreeNormalizer.h>
%include <shogun/kernel/MultitaskKernelMaskNormalizer.h>
%include <shogun/kernel/MultitaskKernelMaskPairNormalizer.h>
%include <shogun/kernel/MultitaskKernelPlifNormalizer.h>
%include <shogun/kernel/OligoStringKernel.h>
%include <shogun/kernel/PolyKernel.h>
%include <shogun/kernel/PolyMatchStringKernel.h>
%include <shogun/kernel/PowerKernel.h>
%include <shogun/kernel/LogKernel.h>
%include <shogun/kernel/GaussianMatchStringKernel.h>
%include <shogun/kernel/SNPStringKernel.h>
%include <shogun/kernel/RegulatoryModulesStringKernel.h>
%include <shogun/kernel/PolyMatchWordStringKernel.h>
%include <shogun/kernel/SalzbergWordStringKernel.h>
%include <shogun/kernel/SigmoidKernel.h>
%include <shogun/kernel/SimpleLocalityImprovedStringKernel.h>
%include <shogun/kernel/SphericalKernel.h>
%include <shogun/kernel/SplineKernel.h>
%include <shogun/kernel/SqrtDiagKernelNormalizer.h>
%include <shogun/kernel/TanimotoKernelNormalizer.h>
%include <shogun/kernel/TensorProductPairKernel.h>
%include <shogun/kernel/TStudentKernel.h>
%include <shogun/kernel/WaveKernel.h>
%include <shogun/kernel/WaveletKernel.h>
%include <shogun/kernel/WeightedCommWordStringKernel.h>
%include <shogun/kernel/WeightedDegreePositionStringKernel.h>
%include <shogun/kernel/WeightedDegreeStringKernel.h>
%include <shogun/kernel/WeightedDegreeRBFKernel.h>
%include <shogun/kernel/SpectrumMismatchRBFKernel.h>
%include <shogun/kernel/ZeroMeanCenterKernelNormalizer.h>
%include <shogun/kernel/MultiquadricKernel.h>
%include <shogun/kernel/RationalQuadraticKernel.h>
