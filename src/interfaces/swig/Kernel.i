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
%shared_ptr(shogun::CombinedKernel)

PROTOCOLS_CUSTOMKERNEL(CustomKernel, float32_t, "f\0", NPY_FLOAT32)
%shared_ptr(shogun::CustomKernel)

%shared_ptr(shogun::DirectorKernel)

#ifdef USE_FLOAT64
    %shared_ptr(shogun::StringKernel<float64_t>)
    %shared_ptr(shogun::SparseKernel<float64_t>)
#endif
#ifdef USE_UINT16
    %shared_ptr(shogun::StringKernel<uint16_t>)
    %shared_ptr(shogun::SparseKernel<uint16_t>)
#endif
#ifdef USE_CHAR
    %shared_ptr(shogun::StringKernel<char>)
#endif
#ifdef USE_UINT64
    %shared_ptr(shogun::StringKernel<uint64_t>)
#endif

/* Include Class Headers to make them visible from within the target language */
%include <shogun/kernel/Kernel.h>

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
#ifdef USE_UINT16
    %template(StringWordKernel) StringKernel<uint16_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharKernel) StringKernel<char>;
#endif
#ifdef USE_UINT64
    %template(StringUlongKernel) StringKernel<uint64_t>;
#endif
}

%include <shogun/kernel/normalizer/KernelNormalizer.h>
%include <shogun/kernel/CombinedKernel.h>
%include <shogun/kernel/CustomKernel.h>
%include <shogun/kernel/DirectorKernel.h>

EXTEND_CUSTOMKERNEL(CustomKernel, float32_t, NPY_FLOAT32)
