#include <shogun/preprocessor/SimplePreprocessor.h>

namespace shogun
{
template <class ST> 
CSimplePreprocessor<ST>::CSimplePreprocessor() : CPreprocessor()
{
}

template<> EFeatureType CSimplePreprocessor<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<> EFeatureType CSimplePreprocessor<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> EFeatureType CSimplePreprocessor<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> EFeatureType CSimplePreprocessor<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<> EFeatureType CSimplePreprocessor<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> EFeatureType CSimplePreprocessor<char>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType CSimplePreprocessor<int8_t>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType CSimplePreprocessor<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> EFeatureType CSimplePreprocessor<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> EFeatureType CSimplePreprocessor<uint32_t>::get_feature_type()
{
	return F_UINT;
}


template<> EFeatureType CSimplePreprocessor<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> EFeatureType CSimplePreprocessor<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> EFeatureType CSimplePreprocessor<bool>::get_feature_type()
{
	return F_BOOL;
}

template <class ST>
EFeatureClass CSimplePreprocessor<ST>::get_feature_class()
{
	return C_SIMPLE;
}

template <class ST>
EPreprocessorType CSimplePreprocessor<ST>::get_type() const
{
	return P_UNKNOWN;
}

template class CSimplePreprocessor<bool>;
template class CSimplePreprocessor<char>;
template class CSimplePreprocessor<int8_t>;
template class CSimplePreprocessor<uint8_t>;
template class CSimplePreprocessor<int16_t>;
template class CSimplePreprocessor<uint16_t>;
template class CSimplePreprocessor<int32_t>;
template class CSimplePreprocessor<uint32_t>;
template class CSimplePreprocessor<int64_t>;
template class CSimplePreprocessor<uint64_t>;
template class CSimplePreprocessor<float32_t>;
template class CSimplePreprocessor<float64_t>;
template class CSimplePreprocessor<floatmax_t>;
}
