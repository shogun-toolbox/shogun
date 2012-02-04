/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
/* Remove C Prefix */
%rename(Preprocessor) CPreprocessor;
%rename(SparsePreprocessor) CSparsePreprocessor;
%rename(SumOne) CSumOne;
%rename(NormOne) CNormOne;
%rename(LogPlusOne) CLogPlusOne;
%rename(PruneVarSubMean) CPruneVarSubMean;
%rename(RandomFourierGaussPreproc) CRandomFourierGaussPreproc;

%rename(DimensionReductionPreprocessor) CDimensionReductionPreprocessor;
%rename(PCA) CPCA;
%rename(KernelPCA) CKernelPCA;

%rename(SortUlongString) CSortUlongString;
%rename(SortWordString) CSortWordString;

%newobject shogun::CKernelPCA::apply_to_string_features;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>
%include <shogun/preprocessor/Preprocessor.h>

/* Templates Class SimplePreprocessor*/
%include <shogun/preprocessor/SimplePreprocessor.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealPreprocessor) CSimplePreprocessor<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongPreprocessor) CSimplePreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordPreprocessor) CSimplePreprocessor<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortPreprocessor) CSimplePreprocessor<int16_t>;
#endif
#ifdef USE_UINT8
    %template(BytePreprocessor) CSimplePreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharPreprocessor) CSimplePreprocessor<char>;
#endif
}

/* Templates Class DimensionReductionPreprocessor */
%include <shogun/preprocessor/DimensionReductionPreprocessor.h>

/* Templates Class StringPreprocessor*/
%include <shogun/preprocessor/StringPreprocessor.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(StringUlongPreprocessor) CStringPreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordPreprocessor) CStringPreprocessor<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(StringBytePreprocessor) CStringPreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharPreprocessor) CStringPreprocessor<char>;
#endif
}

/* Templates Class DecompressString*/
%include <shogun/preprocessor/DecompressString.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(DecompressUlongString) CDecompressString<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(DecompressWordString) CDecompressString<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(DecompressByteString) CDecompressString<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(DecompressCharString) CDecompressString<char>;
#endif
}
%include <shogun/preprocessor/SparsePreprocessor.h>
%include <shogun/preprocessor/NormOne.h>
%include <shogun/preprocessor/SumOne.h>
%include <shogun/preprocessor/LogPlusOne.h>
%include <shogun/preprocessor/PruneVarSubMean.h>
%include <shogun/preprocessor/RandomFourierGaussPreproc.h>

%include <shogun/preprocessor/PCA.h>
%include <shogun/preprocessor/KernelPCA.h>

%include <shogun/preprocessor/SortUlongString.h>
%include <shogun/preprocessor/SortWordString.h>

