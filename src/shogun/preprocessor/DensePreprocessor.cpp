#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun
{
template <class ST>
CDensePreprocessor<ST>::CDensePreprocessor() : CPreprocessor()
{
}

template<> EFeatureType CDensePreprocessor<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<> EFeatureType CDensePreprocessor<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> EFeatureType CDensePreprocessor<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> EFeatureType CDensePreprocessor<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<> EFeatureType CDensePreprocessor<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> EFeatureType CDensePreprocessor<char>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType CDensePreprocessor<int8_t>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType CDensePreprocessor<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> EFeatureType CDensePreprocessor<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> EFeatureType CDensePreprocessor<uint32_t>::get_feature_type()
{
	return F_UINT;
}


template<> EFeatureType CDensePreprocessor<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> EFeatureType CDensePreprocessor<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> EFeatureType CDensePreprocessor<bool>::get_feature_type()
{
	return F_BOOL;
}

template <class ST>
EFeatureClass CDensePreprocessor<ST>::get_feature_class()
{
	return C_DENSE;
}

template <class ST>
EPreprocessorType CDensePreprocessor<ST>::get_type() const
{
	return P_UNKNOWN;
}

template <class ST>
CFeatures* CDensePreprocessor<ST>::apply(CFeatures* features)
{
	REQUIRE(features->get_feature_class()==C_DENSE, "Provided features (%d) "
			"has to be of C_DENSE (%d) class!\n",
			features->get_feature_class(), C_DENSE);

	SGMatrix<ST> feat_matrix=apply_to_feature_matrix(features);
	CDenseFeatures<ST>* preprocessed=new CDenseFeatures<ST>(feat_matrix);
	SG_REF(preprocessed);
	return preprocessed;
}

template class CDensePreprocessor<bool>;
template class CDensePreprocessor<char>;
template class CDensePreprocessor<int8_t>;
template class CDensePreprocessor<uint8_t>;
template class CDensePreprocessor<int16_t>;
template class CDensePreprocessor<uint16_t>;
template class CDensePreprocessor<int32_t>;
template class CDensePreprocessor<uint32_t>;
template class CDensePreprocessor<int64_t>;
template class CDensePreprocessor<uint64_t>;
template class CDensePreprocessor<float32_t>;
template class CDensePreprocessor<float64_t>;
template class CDensePreprocessor<floatmax_t>;
}
