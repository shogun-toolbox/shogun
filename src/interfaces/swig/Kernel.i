/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
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

%rename(AvgDiagKernelNormalizer) CAvgDiagKernelNormalizer;
%rename(RidgeKernelNormalizer) CRidgeKernelNormalizer;
%rename(CommUlongStringKernel) CCommUlongStringKernel;
%rename(CommWordStringKernel) CCommWordStringKernel;

PROTOCOLS_CUSTOMKERNEL(CustomKernel, float32_t, "f\0", NPY_FLOAT32)
%rename(CustomKernel) CCustomKernel;

%rename(DiagKernel) CDiagKernel;
#ifdef USE_GPL_SHOGUN
%rename(DistantSegmentsKernel) CDistantSegmentsKernel;
#endif //USE_GPL_SHOGUN
%rename(DiceKernelNormalizer) CDiceKernelNormalizer;
%rename(ScatterKernelNormalizer) CScatterKernelNormalizer;
%rename(VarianceKernelNormalizer) CVarianceKernelNormalizer;
%rename(FixedDegreeStringKernel) CFixedDegreeStringKernel;
%rename(DirectorKernel) CDirectorKernel;
%rename(HistogramWordStringKernel) CHistogramWordStringKernel;
%rename(IdentityKernelNormalizer) CIdentityKernelNormalizer;
%rename(LinearStringKernel) CLinearStringKernel;
%rename(SparseSpatialSampleStringKernel) CSparseSpatialSampleStringKernel;
%rename(LocalAlignmentStringKernel) CLocalAlignmentStringKernel;
%rename(LocalityImprovedStringKernel) CLocalityImprovedStringKernel;
%rename(MatchWordStringKernel) CMatchWordStringKernel;
%rename(OligoStringKernel) COligoStringKernel;
%rename(PolyMatchStringKernel) CPolyMatchStringKernel;
%rename(GaussianMatchStringKernel) CGaussianMatchStringKernel;
%rename(SNPStringKernel) CSNPStringKernel;
%rename(RegulatoryModulesStringKernel) CRegulatoryModulesStringKernel;
%rename(PolyMatchWordStringKernel) CPolyMatchWordStringKernel;
%rename(SalzbergWordStringKernel) CSalzbergWordStringKernel;
%rename(SimpleLocalityImprovedStringKernel) CSimpleLocalityImprovedStringKernel;
%rename(SqrtDiagKernelNormalizer) CSqrtDiagKernelNormalizer;
%rename(TanimotoKernelNormalizer) CTanimotoKernelNormalizer;
%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;
%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;
%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;
%rename(ZeroMeanCenterKernelNormalizer) CZeroMeanCenterKernelNormalizer;
%rename(SubsequenceStringKernel) CSubsequenceStringKernel;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/kernel/Kernel.h>

%include <shogun/kernel/DotKernel.h>

/** Instantiate RandomMixin */
%template(RandomMixinDotKernel) shogun::RandomMixin<shogun::CDotKernel, std::mt19937_64>;

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
%include <shogun/kernel/string/CommUlongStringKernel.h>
%include <shogun/kernel/string/CommWordStringKernel.h>
%include <shogun/kernel/CombinedKernel.h>
%include <shogun/kernel/CustomKernel.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/kernel/string/DistantSegmentsKernel.h>
#endif //USE_GPL_SHOGUN
%include <shogun/kernel/string/FixedDegreeStringKernel.h>
%include <shogun/kernel/DirectorKernel.h>
%include <shogun/kernel/string/HistogramWordStringKernel.h>
%include <shogun/kernel/string/LinearStringKernel.h>
%include <shogun/kernel/string/SparseSpatialSampleStringKernel.h>
%include <shogun/kernel/string/LocalAlignmentStringKernel.h>
%include <shogun/kernel/string/LocalityImprovedStringKernel.h>
%include <shogun/kernel/string/MatchWordStringKernel.h>
%include <shogun/kernel/string/OligoStringKernel.h>
%include <shogun/kernel/string/PolyMatchStringKernel.h>
%include <shogun/kernel/string/GaussianMatchStringKernel.h>
%include <shogun/kernel/string/SNPStringKernel.h>
%include <shogun/kernel/string/RegulatoryModulesStringKernel.h>
%include <shogun/kernel/string/PolyMatchWordStringKernel.h>
%include <shogun/kernel/string/SalzbergWordStringKernel.h>
%include <shogun/kernel/string/SimpleLocalityImprovedStringKernel.h>
%include <shogun/kernel/string/WeightedCommWordStringKernel.h>
%include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
%include <shogun/kernel/string/WeightedDegreeStringKernel.h>
%include <shogun/kernel/string/SpectrumMismatchRBFKernel.h>
%include <shogun/kernel/string/SubsequenceStringKernel.h>

EXTEND_CUSTOMKERNEL(CustomKernel, float32_t, NPY_FLOAT32)
