#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun
{
template <class ST>
CDensePreprocessor<ST>::CDensePreprocessor() : CPreprocessor()
{
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<char>::get_feature_type()
{
	return F_CHAR;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<int8_t>::get_feature_type()
{
	return F_CHAR;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<uint32_t>::get_feature_type()
{
	return F_UINT;
}


template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> SHOGUN_EXPORT EFeatureType CDensePreprocessor<bool>::get_feature_type()
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
CFeatures* CDensePreprocessor<ST>::transform(CFeatures* features, bool inplace)
{
	REQUIRE(features->get_feature_class()==C_DENSE, "Provided features (%d) "
			"has to be of C_DENSE (%d) class!\n",
			features->get_feature_class(), C_DENSE);

	SG_REF(features);
	auto matrix = features->as<CDenseFeatures<ST>>()->get_feature_matrix();
	if (!inplace)
		matrix = matrix.clone();
	auto feat_matrix = apply_to_matrix(matrix);
	auto preprocessed = new CDenseFeatures<ST>(feat_matrix);

	SG_UNREF(features);
	return preprocessed;
}

template <class ST>
CFeatures*
CDensePreprocessor<ST>::inverse_transform(CFeatures* features, bool inplace)
{
	REQUIRE(
		features->get_feature_class() == C_DENSE,
		"Provided features (%d) "
		"has to be of C_DENSE (%d) class!\n",
		features->get_feature_class(), C_DENSE);

	SG_REF(features);
	auto matrix = features->as<CDenseFeatures<ST>>()->get_feature_matrix();
	if (!inplace)
		matrix = matrix.clone();
	auto feat_matrix = inverse_apply_to_matrix(matrix);
	auto preprocessed = new CDenseFeatures<ST>(feat_matrix);

	SG_UNREF(features);
	return preprocessed;
}

template <class ST>
SGMatrix<ST>
CDensePreprocessor<ST>::inverse_apply_to_matrix(SGMatrix<ST> matrix)
{
	SG_SNOTIMPLEMENTED;

	return SGMatrix<ST>();
}

template class SHOGUN_EXPORT CDensePreprocessor<bool>;
template class SHOGUN_EXPORT CDensePreprocessor<char>;
template class SHOGUN_EXPORT CDensePreprocessor<int8_t>;
template class SHOGUN_EXPORT CDensePreprocessor<uint8_t>;
template class SHOGUN_EXPORT CDensePreprocessor<int16_t>;
template class SHOGUN_EXPORT CDensePreprocessor<uint16_t>;
template class SHOGUN_EXPORT CDensePreprocessor<int32_t>;
template class SHOGUN_EXPORT CDensePreprocessor<uint32_t>;
template class SHOGUN_EXPORT CDensePreprocessor<int64_t>;
template class SHOGUN_EXPORT CDensePreprocessor<uint64_t>;
template class SHOGUN_EXPORT CDensePreprocessor<float32_t>;
template class SHOGUN_EXPORT CDensePreprocessor<float64_t>;
template class SHOGUN_EXPORT CDensePreprocessor<floatmax_t>;
}
