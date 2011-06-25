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
"The `Distance` module gathers all distances available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distance
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Distance_doxygen.i"
#endif
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_distance_matrix(self) -> numpy 2dim array of float") get_distance_matrix;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Features_includes.i"
%include "Distance_includes.i"
%include "Preprocessor_includes.i"

%import "Features.i"

/* Remove C Prefix */
%rename(BaseDistance) CDistance;
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
%rename(EuclidianDistance) CEuclidianDistance;
%rename(SparseEuclidianDistance) CSparseEuclidianDistance;
%rename(BrayCurtisDistance) CBrayCurtisDistance;
%rename(ChiSquareDistance) CChiSquareDistance;
%rename(CosineDistance) CCosineDistance;
%rename(TanimotoDistance) CTanimotoDistance;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distance/Distance.h>
%include <shogun/distance/CustomDistance.h>
%include <shogun/distance/KernelDistance.h>

/* Templates Class SimpleDistance*/
%include <shogun/distance/SimpleDistance.h>
namespace shogun
{
    %template(SimpleRealDistance) CSimpleDistance<float64_t>;
    %template(SimpleWordDistance) CSimpleDistance<uint16_t>;
    %template(SimpleCharDistance) CSimpleDistance<char>;
    %template(SimpleIntDistance) CSimpleDistance<int32_t>;
}

/* Templates Class SparseDistance*/
%include <shogun/distance/SparseDistance.h>
namespace shogun
{
    %template(SparseRealDistance) CSparseDistance<float64_t>;
    %template(SparseWordDistance) CSparseDistance<uint16_t>;
    %template(SparseCharDistance) CSparseDistance<char>;
    %template(SparseIntDistance) CSparseDistance<int32_t>;
}

/* Templates Class StringDistance*/
%include <shogun/distance/StringDistance.h>
namespace shogun
{
    %template(StringRealDistance) CStringDistance<float64_t>;
    %template(StringWordDistance) CStringDistance<uint16_t>;
    %template(StringCharDistance) CStringDistance<char>;
    %template(StringIntDistance) CStringDistance<int32_t>;
    %template(StringUlongDistance) CStringDistance<uint64_t>;
}

%include <shogun/features/FeatureTypes.h>
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
%include <shogun/distance/EuclidianDistance.h>
%include <shogun/distance/SparseEuclidianDistance.h>
%include <shogun/distance/BrayCurtisDistance.h>
%include <shogun/distance/ChiSquareDistance.h>
%include <shogun/distance/CosineDistance.h>
%include <shogun/distance/TanimotoDistance.h>
