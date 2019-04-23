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
%feature("director") shogun::DirectorKernel;
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
%shared_ptr(shogun::Kernel)
%shared_ptr(shogun::KernelNormalizer)
%shared_ptr(shogun::PyramidChi2)
%shared_ptr(shogun::ANOVAKernel)
%shared_ptr(shogun::AUCKernel)
%shared_ptr(shogun::BesselKernel)
%shared_ptr(shogun::AvgDiagKernelNormalizer)
%shared_ptr(shogun::RidgeKernelNormalizer)
%shared_ptr(shogun::CircularKernel)
%shared_ptr(shogun::Chi2Kernel)
%shared_ptr(shogun::CombinedKernel)
%shared_ptr(shogun::ProductKernel)
%shared_ptr(shogun::CommUlongStringKernel)
%shared_ptr(shogun::CommWordStringKernel)
%shared_ptr(shogun::ConstKernel)

PROTOCOLS_CUSTOMKERNEL(CustomKernel, float32_t, "f\0", NPY_FLOAT32)
%shared_ptr(shogun::CustomKernel)

%shared_ptr(shogun::DiagKernel)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::DistantSegmentsKernel)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::WaveKernel)
%shared_ptr(shogun::CauchyKernel)
%shared_ptr(shogun::DiceKernelNormalizer)
%shared_ptr(shogun::ExponentialKernel)
%shared_ptr(shogun::ScatterKernelNormalizer)
%shared_ptr(shogun::VarianceKernelNormalizer)
%shared_ptr(shogun::DistanceKernel)
%shared_ptr(shogun::FixedDegreeStringKernel)
%shared_ptr(shogun::ShiftInvariantKernel)
%shared_ptr(shogun::GaussianCompactKernel)
%shared_ptr(shogun::DirectorKernel)
%shared_ptr(shogun::WaveletKernel)
%shared_ptr(shogun::GaussianShiftKernel)
%shared_ptr(shogun::GaussianShortRealKernel)
%shared_ptr(shogun::HistogramIntersectionKernel)
%shared_ptr(shogun::HistogramWordStringKernel)
%shared_ptr(shogun::IdentityKernelNormalizer)
%shared_ptr(shogun::InverseMultiQuadricKernel)
%shared_ptr(shogun::LinearKernel)
%shared_ptr(shogun::LinearStringKernel)
%shared_ptr(shogun::SparseSpatialSampleStringKernel)
%shared_ptr(shogun::SplineKernel)
%shared_ptr(shogun::LocalAlignmentStringKernel)
%shared_ptr(shogun::LocalityImprovedStringKernel)
%shared_ptr(shogun::MatchWordStringKernel)
%shared_ptr(shogun::OligoStringKernel)
%shared_ptr(shogun::PolyMatchStringKernel)
%shared_ptr(shogun::PowerKernel)
%shared_ptr(shogun::LogKernel)
%shared_ptr(shogun::GaussianMatchStringKernel)
%shared_ptr(shogun::SNPStringKernel)
%shared_ptr(shogun::RegulatoryModulesStringKernel)
%shared_ptr(shogun::PolyMatchWordStringKernel)
%shared_ptr(shogun::SalzbergWordStringKernel)
%shared_ptr(shogun::SigmoidKernel)
%shared_ptr(shogun::SphericalKernel)
%shared_ptr(shogun::SimpleLocalityImprovedStringKernel)
%shared_ptr(shogun::SqrtDiagKernelNormalizer)
%shared_ptr(shogun::TanimotoKernelNormalizer)
%shared_ptr(shogun::TensorProductPairKernel)
%shared_ptr(shogun::TStudentKernel)
%shared_ptr(shogun::WeightedCommWordStringKernel)
%shared_ptr(shogun::WeightedDegreePositionStringKernel)
%shared_ptr(shogun::WeightedDegreeStringKernel)
%shared_ptr(shogun::WeightedDegreeRBFKernel)
%shared_ptr(shogun::SpectrumMismatchRBFKernel)
%shared_ptr(shogun::ZeroMeanCenterKernelNormalizer)
%shared_ptr(shogun::DotKernel)
%shared_ptr(shogun::RationalQuadraticKernel)
%shared_ptr(shogun::MultiquadricKernel)
%shared_ptr(shogun::JensenShannonKernel)

%shared_ptr(shogun::ExponentialARDKernel)
%shared_ptr(shogun::GaussianARDKernel)

%shared_ptr(shogun::GaussianARDSparseKernel)

%shared_ptr(shogun::SubsequenceStringKernel)
%shared_ptr(shogun::PeriodicKernel)

#ifdef USE_FLOAT64
    %shared_ptr(shogun::StringKernel<float64_t>)
    %shared_ptr(shogun::SparseKernel<float64_t>)
#endif
#ifdef USE_INT16
    %shared_ptr(shogun::StringKernel<int16_t>)
#endif
#ifdef USE_UINT16
    %shared_ptr(shogun::StringKernel<uint16_t>)
    %shared_ptr(shogun::SparseKernel<uint16_t>)
#endif
#ifdef USE_CHAR
    %shared_ptr(shogun::StringKernel<char>)
#endif
#ifdef USE_UINT32
    %shared_ptr(shogun::StringKernel<int32_t>)
#endif
#ifdef USE_UINT64
    %shared_ptr(shogun::StringKernel<uint64_t>)
#endif
#ifdef USE_UINT8
    %shared_ptr(shogun::StringKernel<uint8_t>)
#endif

/* Include Class Headers to make them visible from within the target language */
%include <shogun/kernel/Kernel.h>

%include <shogun/kernel/DotKernel.h>

/* Templated Class SparseKernel */
%include <shogun/kernel/SparseKernel.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(SparseRealKernel) SparseKernel<float64_t>;
#endif
#ifdef USE_UINT16
    %template(SparseWordKernel) SparseKernel<uint16_t>;
#endif
}

/* Templated Class StringKernel */
%include <shogun/kernel/string/StringKernel.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(StringRealKernel) StringKernel<float64_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordKernel) StringKernel<uint16_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharKernel) StringKernel<char>;
#endif
#ifdef USE_UINT32
    %template(StringIntKernel) StringKernel<int32_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongKernel) StringKernel<uint64_t>;
#endif
#ifdef USE_INT16
    %template(StringShortKernel) StringKernel<int16_t>;
#endif
#ifdef USE_UINT8
    %template(StringByteKernel) StringKernel<uint8_t>;
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
