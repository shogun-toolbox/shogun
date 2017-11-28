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
%rename(KernelDistance) CKernelDistance;
%rename(RealDistance) CRealDistance;
%rename(CanberraMetric) CCanberraMetric;
%rename(ChebyshewMetric) CChebyshewMetric;
%rename(GeodesicMetric) CGeodesicMetric;
%rename(JensenMetric) CJensenMetric;
%rename(ManhattanMetric) CManhattanMetric;
%rename(MinkowskiMetric) CMinkowskiMetric;
%rename(HammingWordDistance) CHammingWordDistance;
%rename(ManhattanWordDistance) CManhattanWordDistance;
%rename(CanberraWordDistance) CCanberraWordDistance;
%rename(EuclideanDistance) CEuclideanDistance;
%rename(SparseEuclideanDistance) CSparseEuclideanDistance;
%rename(BrayCurtisDistance) CBrayCurtisDistance;
%rename(ChiSquareDistance) CChiSquareDistance;
%rename(CosineDistance) CCosineDistance;
%rename(TanimotoDistance) CTanimotoDistance;
%rename(MahalanobisDistance) CMahalanobisDistance;
%rename(DirectorDistance) CDirectorDistance;
%rename(CustomMahalanobisDistance) CCustomMahalanobisDistance;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distance/Distance.h>
%include <shogun/distance/CustomDistance.h>
%include <shogun/distance/KernelDistance.h>

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
%include <shogun/distance/CanberraMetric.h>
%include <shogun/distance/ChebyshewMetric.h>
%include <shogun/distance/GeodesicMetric.h>
%include <shogun/distance/JensenMetric.h>
%include <shogun/distance/ManhattanMetric.h>
%include <shogun/distance/MinkowskiMetric.h>
%include <shogun/distance/HammingWordDistance.h>
%include <shogun/distance/ManhattanWordDistance.h>
%include <shogun/distance/CanberraWordDistance.h>
%include <shogun/distance/EuclideanDistance.h>
%include <shogun/distance/SparseEuclideanDistance.h>
%include <shogun/distance/BrayCurtisDistance.h>
%include <shogun/distance/ChiSquareDistance.h>
%include <shogun/distance/CosineDistance.h>
%include <shogun/distance/TanimotoDistance.h>
%include <shogun/distance/MahalanobisDistance.h>
%include <shogun/distance/DirectorDistance.h>
%include <shogun/distance/CustomMahalanobisDistance.h>
