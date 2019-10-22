/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
/* Feature selection framework */
%shared_ptr(shogun::Preprocessor)
#%shared_ptr(shogun::DependenceMaximization)
#%shared_ptr(shogun::KernelDependenceMaximization)

#ifdef USE_UINT64
%shared_ptr(shogun::StringPreprocessor<uint64_t>)
%shared_ptr(shogun::DecompressString<uint64_t>)
%shared_ptr(shogun::FeatureSelection<uint64_t>)
#endif
#ifdef USE_UINT16
%shared_ptr(shogun::StringPreprocessor<uint16_t>)
%shared_ptr(shogun::DecompressString<uint16_t>)
%shared_ptr(shogun::FeatureSelection<uint16_t>)
#endif
#ifdef USE_UINT8
%shared_ptr(shogun::StringPreprocessor<uint8_t>)
%shared_ptr(shogun::DecompressString<uint8_t>)
%shared_ptr(shogun::FeatureSelection<uint8_t>)
#endif
#ifdef USE_CHAR
%shared_ptr(shogun::StringPreprocessor<char>)
%shared_ptr(shogun::DecompressString<char>)
%shared_ptr(shogun::FeatureSelection<char>)
#endif
#ifdef USE_FLOAT64
%shared_ptr(shogun::FeatureSelection<float64_t>)
#endif
#ifdef USE_INT16
%shared_ptr(shogun::FeatureSelection<int16_t>)
#endif


/* Include Class Headers to make them visible from within the target language */
%include <shogun/preprocessor/Preprocessor.h>

/* Templates Class StringPreprocessor*/
%include <shogun/preprocessor/StringPreprocessor.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(StringUlongPreprocessor) StringPreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordPreprocessor) StringPreprocessor<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(StringBytePreprocessor) StringPreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharPreprocessor) StringPreprocessor<char>;
#endif
}

/* Templates Class DecompressString*/
%include <shogun/preprocessor/DecompressString.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(DecompressUlongString) DecompressString<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(DecompressWordString) DecompressString<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(DecompressByteString) DecompressString<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(DecompressCharString) DecompressString<char>;
#endif
}

/* Templates Class FeatureSelection */
%include <shogun/preprocessor/FeatureSelection.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealFeatureSelection) FeatureSelection<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongFeatureSelection) FeatureSelection<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordFeatureSelection) FeatureSelection<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortFeatureSelection) FeatureSelection<int16_t>;
#endif
#ifdef USE_UINT8
    %template(ByteFeatureSelection) FeatureSelection<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharFeatureSelection) FeatureSelection<char>;
#endif
}

