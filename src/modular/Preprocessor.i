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
"The `Preprocessor` module gathers all preprocessors available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Preprocessor

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Preprocessor_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/lib/Compressor.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/NormDerivativeLem3.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/PCACut.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/preprocessor/KernelPCACut.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/DecompressString.h>
#include <shogun/preprocessor/SortUlongString.h>
#include <shogun/preprocessor/SortWordString.h>
#include <shogun/preprocessor/SparsePreprocessor.h>
#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <shogun/preprocessor/ClassicMDS.h>
%}

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num_feat)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num_new_dim)};

/* Remove C Prefix */
%rename(BasePreprocessor) CPreprocessor;
%rename(LogPlusOne) CLogPlusOne;
%rename(NormDerivativeLem3) CNormDerivativeLem3;
%rename(NormOne) CNormOne;
%rename(PCACut) CPCACut;
%rename(Kernel) CKernel;
%rename(KernelPCACut) CKernelPCACut;
%rename(PruneVarSubMean) CPruneVarSubMean;
%rename(SortUlongString) CSortUlongString;
%rename(SortWordString) CSortWordString;
%rename(SparsePreprocessor) CSparsePreprocessor;
%rename(RandomFourierGaussPreproc) CRandomFourierGaussPreproc;
%rename(ClassicMDS) CClassicMDS;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>
%include <shogun/features/FeatureTypes.h>
%include <shogun/preprocessor/Preprocessor.h>

/* Templates Class SimplePreprocessor*/
%include <shogun/preprocessor/SimplePreprocessor.h>
namespace shogun
{
    %template(SimpleRealPreprocessor) CSimplePreprocessor<float64_t>;
    %template(SimpleUlongPreprocessor) CSimplePreprocessor<uint64_t>;
    %template(SimpleWordPreprocessor) CSimplePreprocessor<uint16_t>;
    %template(SimpleShortPreprocessor) CSimplePreprocessor<int16_t>;
    %template(SimpleBytePreprocessor) CSimplePreprocessor<uint8_t>;
    %template(SimpleCharPreprocessor) CSimplePreprocessor<char>;
}

/* Templates Class StringPreprocessor*/
%include <shogun/preprocessor/StringPreprocessor.h>
namespace shogun
{
    %template(StringUlongPreprocessor) CStringPreprocessor<uint64_t>;
    %template(StringWordPreprocessor) CStringPreprocessor<uint16_t>;
    %template(StringBytePreprocessor) CStringPreprocessor<uint8_t>;
    %template(StringCharPreprocessor) CStringPreprocessor<char>;
}

/* Templates Class DecompressString*/
%include <shogun/preprocessor/DecompressString.h>
namespace shogun
{
    %template(DecompressUlongString) CDecompressString<uint64_t>;
    %template(DecompressWordString) CDecompressString<uint16_t>;
    %template(DecompressByteString) CDecompressString<uint8_t>;
    %template(DecompressCharString) CDecompressString<char>;
}

%include <shogun/preprocessor/LogPlusOne.h>
%include <shogun/preprocessor/NormDerivativeLem3.h>
%include <shogun/preprocessor/NormOne.h>
%include <shogun/preprocessor/PCACut.h>
%include <shogun/kernel/Kernel.h>
%include <shogun/preprocessor/KernelPCACut.h>
%include <shogun/preprocessor/PruneVarSubMean.h>
%include <shogun/preprocessor/SortUlongString.h>
%include <shogun/preprocessor/SortWordString.h>
%include <shogun/preprocessor/SparsePreprocessor.h>
%include <shogun/preprocessor/RandomFourierGaussPreproc.h>
%include <shogun/preprocessor/ClassicMDS.h>
