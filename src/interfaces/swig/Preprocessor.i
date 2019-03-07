/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
/* Feature selection framework */
#%rename(DependenceMaximization) CDependenceMaximization;
#%rename(KernelDependenceMaximization) CDependenceMaximization;

%newobject shogun::CFeatureSelection::remove_feats;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>

/* Templates Class DensePreprocessor*/
%include <shogun/preprocessor/DensePreprocessor.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealPreprocessor) CDensePreprocessor<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongPreprocessor) CDensePreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordPreprocessor) CDensePreprocessor<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortPreprocessor) CDensePreprocessor<int16_t>;
#endif
#ifdef USE_UINT8
    %template(BytePreprocessor) CDensePreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharPreprocessor) CDensePreprocessor<char>;
#endif
}


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

/* Templates Class FeatureSelection */
%include <shogun/preprocessor/FeatureSelection.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealFeatureSelection) CFeatureSelection<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongFeatureSelection) CFeatureSelection<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordFeatureSelection) CFeatureSelection<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortFeatureSelection) CFeatureSelection<int16_t>;
#endif
#ifdef USE_UINT8
    %template(ByteFeatureSelection) CFeatureSelection<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharFeatureSelection) CFeatureSelection<char>;
#endif
}

