/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_distance_matrix(self) -> numpy 2dim array of float") get_distance_matrix;
#endif

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorDistance;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

/* Remove C Prefix */
%shared_ptr(shogun::Distance)
%shared_ptr(shogun::CustomDistance)
%shared_ptr(shogun::RealDistance)
#ifdef USE_CHAR
    %shared_ptr(shogun::DenseDistance<char>)
    %shared_ptr(shogun::SparseDistance<char>)
    %shared_ptr(shogun::StringDistance<char>)
#endif
#ifdef USE_UINT16
    %shared_ptr(shogun::DenseDistance<uint16_t>)
    %shared_ptr(shogun::SparseDistance<uint16_t>)
    %shared_ptr(shogun::StringDistance<uint16_t>)
#endif
#ifdef USE_INT32
    %shared_ptr(shogun::DenseDistance<int32_t>)
    %shared_ptr(shogun::SparseDistance<int32_t>)
    %shared_ptr(shogun::StringDistance<int32_t>)
#endif
#ifdef USE_UINT64
    %shared_ptr(shogun::StringDistance<uint64_t>)
#endif
#ifdef USE_FLOAT64
    %shared_ptr(shogun::DenseDistance<float64_t>)
    %shared_ptr(shogun::SparseDistance<float64_t>)
    %shared_ptr(shogun::StringDistance<float64_t>)
#endif


/* Include Class Headers to make them visible from within the target language */
%include <shogun/distance/Distance.h>
%include <shogun/distance/CustomDistance.h>

/* Templates Class DenseDistance*/
%include <shogun/distance/DenseDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(DenseCharDistance) DenseDistance<char>;
#endif
#ifdef USE_UINT16
    %template(DenseWordDistance) DenseDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(DenseIntDistance) DenseDistance<int32_t>;
#endif
#ifdef USE_FLOAT64
    %template(DenseRealDistance) DenseDistance<float64_t>;
#endif

}

/* Templates Class SparseDistance*/
%include <shogun/distance/SparseDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(SparseCharDistance) SparseDistance<char>;
#endif
#ifdef USE_UINT16
    %template(SparseWordDistance) SparseDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(SparseIntDistance) SparseDistance<int32_t>;
#endif
#ifdef USE_FLOAT64
    %template(SparseRealDistance) SparseDistance<float64_t>;
#endif
}

/* Templates Class StringDistance*/
%include <shogun/distance/StringDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(StringCharDistance) StringDistance<char>;
#endif
#ifdef USE_UINT16
    %template(StringWordDistance) StringDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringIntDistance) StringDistance<int32_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongDistance) StringDistance<uint64_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringRealDistance) StringDistance<float64_t>;
#endif
}

%include <shogun/distance/RealDistance.h>
%include <shogun/distance/DirectorDistance.h>
