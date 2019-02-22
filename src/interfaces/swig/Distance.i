/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_distance_matrix(self) -> numpy 2dim array of float") get_distance_matrix;
#endif

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorDistance;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

/* Remove C Prefix */
%rename(Distance) CDistance;
%rename(CustomDistance) CCustomDistance;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distance/Distance.h>
%include <shogun/distance/CustomDistance.h>

/* Templates Class DenseDistance*/
%include <shogun/distance/DenseDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(DenseCharDistance) CDenseDistance<char>;
#endif
#ifdef USE_UINT16
    %template(DenseWordDistance) CDenseDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(DenseIntDistance) CDenseDistance<int32_t>;
#endif
#ifdef USE_FLOAT64
    %template(DenseRealDistance) CDenseDistance<float64_t>;
#endif

}

/* Templates Class SparseDistance*/
%include <shogun/distance/SparseDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(SparseCharDistance) CSparseDistance<char>;
#endif
#ifdef USE_UINT16
    %template(SparseWordDistance) CSparseDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(SparseIntDistance) CSparseDistance<int32_t>;
#endif
#ifdef USE_FLOAT64
    %template(SparseRealDistance) CSparseDistance<float64_t>;
#endif
}

/* Templates Class StringDistance*/
%include <shogun/distance/StringDistance.h>
namespace shogun
{
#ifdef USE_CHAR
    %template(StringCharDistance) CStringDistance<char>;
#endif
#ifdef USE_UINT16
    %template(StringWordDistance) CStringDistance<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringIntDistance) CStringDistance<int32_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongDistance) CStringDistance<uint64_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringRealDistance) CStringDistance<float64_t>;
#endif
}

%include <shogun/distance/RealDistance.h>
%include <shogun/distance/DirectorDistance.h>
