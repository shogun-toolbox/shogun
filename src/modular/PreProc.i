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
"The `PreProc` module gathers all preprocessors available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) PreProc

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "PreProc_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/lib/Compressor.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/preproc/PreProc.h>
#include <shogun/preproc/SimplePreProc.h>
#include <shogun/preproc/StringPreProc.h>
#include <shogun/preproc/LogPlusOne.h>
#include <shogun/preproc/NormDerivativeLem3.h>
#include <shogun/preproc/NormOne.h>
#include <shogun/preproc/PCACut.h>
#include <shogun/preproc/LLE.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/preproc/KernelPCACut.h>
#include <shogun/preproc/PruneVarSubMean.h>
#include <shogun/preproc/DecompressString.h>
#include <shogun/preproc/SortUlongString.h>
#include <shogun/preproc/SortWordString.h>
#include <shogun/preproc/SparsePreProc.h>
#include <shogun/preproc/RandomFourierGaussPreproc.h>
%}

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num_feat)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num_new_dim)};

/* Remove C Prefix */
%rename(BasePreProc) CPreProc;
%rename(LogPlusOne) CLogPlusOne;
%rename(NormDerivativeLem3) CNormDerivativeLem3;
%rename(NormOne) CNormOne;
%rename(PCACut) CPCACut;
%rename(LLE) CLLE;
%rename(Kernel) CKernel;
%rename(KernelPCACut) CKernelPCACut;
%rename(PruneVarSubMean) CPruneVarSubMean;
%rename(SortUlongString) CSortUlongString;
%rename(SortWordString) CSortWordString;
%rename(SparsePreProc) CSparsePreProc;
%rename(RandomFourierGaussPreproc) CRandomFourierGaussPreproc;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>
%include <shogun/features/FeatureTypes.h>
%include <shogun/preproc/PreProc.h>

/* Templates Class SimplePreProc*/
%include <shogun/preproc/SimplePreProc.h>
namespace shogun
{
    %template(SimpleRealPreProc) CSimplePreProc<float64_t>;
    %template(SimpleUlongPreProc) CSimplePreProc<uint64_t>;
    %template(SimpleWordPreProc) CSimplePreProc<uint16_t>;
    %template(SimpleShortPreProc) CSimplePreProc<int16_t>;
    %template(SimpleBytePreProc) CSimplePreProc<uint8_t>;
    %template(SimpleCharPreProc) CSimplePreProc<char>;
}

/* Templates Class StringPreProc*/
%include <shogun/preproc/StringPreProc.h>
namespace shogun
{
    %template(StringUlongPreProc) CStringPreProc<uint64_t>;
    %template(StringWordPreProc) CStringPreProc<uint16_t>;
    %template(StringBytePreProc) CStringPreProc<uint8_t>;
    %template(StringCharPreProc) CStringPreProc<char>;
}

/* Templates Class DecompressString*/
%include <shogun/preproc/DecompressString.h>
namespace shogun
{
    %template(DecompressUlongString) CDecompressString<uint64_t>;
    %template(DecompressWordString) CDecompressString<uint16_t>;
    %template(DecompressByteString) CDecompressString<uint8_t>;
    %template(DecompressCharString) CDecompressString<char>;
}

%include <shogun/preproc/LogPlusOne.h>
%include <shogun/preproc/NormDerivativeLem3.h>
%include <shogun/preproc/NormOne.h>
%include <shogun/preproc/PCACut.h>
%include <shogun/preproc/LLE.h>
%include <shogun/kernel/Kernel.h>
%include <shogun/preproc/KernelPCACut.h>
%include <shogun/preproc/PruneVarSubMean.h>
%include <shogun/preproc/SortUlongString.h>
%include <shogun/preproc/SortWordString.h>
%include <shogun/preproc/SparsePreProc.h>
%include <shogun/preproc/RandomFourierGaussPreproc.h>
