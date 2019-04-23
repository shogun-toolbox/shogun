#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun
{
template <class ST>
DensePreprocessor<ST>::DensePreprocessor() : Preprocessor()
{
}

template<> EFeatureType DensePreprocessor<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<> EFeatureType DensePreprocessor<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> EFeatureType DensePreprocessor<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> EFeatureType DensePreprocessor<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<> EFeatureType DensePreprocessor<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> EFeatureType DensePreprocessor<char>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType DensePreprocessor<int8_t>::get_feature_type()
{
	return F_CHAR;
}

template<> EFeatureType DensePreprocessor<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> EFeatureType DensePreprocessor<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> EFeatureType DensePreprocessor<uint32_t>::get_feature_type()
{
	return F_UINT;
}


template<> EFeatureType DensePreprocessor<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> EFeatureType DensePreprocessor<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> EFeatureType DensePreprocessor<bool>::get_feature_type()
{
	return F_BOOL;
}

template <class ST>
EFeatureClass DensePreprocessor<ST>::get_feature_class()
{
	return C_DENSE;
}

template <class ST>
EPreprocessorType DensePreprocessor<ST>::get_type() const
{
	return P_UNKNOWN;
}

template <class ST>
std::shared_ptr<Features> DensePreprocessor<ST>::transform(std::shared_ptr<Features> features, bool inplace)
{
	REQUIRE(features->get_feature_class()==C_DENSE, "Provided features (%d) "
			"has to be of C_DENSE (%d) class!\n",
			features->get_feature_class(), C_DENSE);

	auto matrix = features->as<DenseFeatures<ST>>()->get_feature_matrix();
	if (!inplace)
		matrix = matrix.clone();
	auto feat_matrix = apply_to_matrix(matrix);
	return std::make_shared<DenseFeatures<ST>>(feat_matrix);
}

template <class ST>
std::shared_ptr<Features>
DensePreprocessor<ST>::inverse_transform(std::shared_ptr<Features> features, bool inplace)
{
	REQUIRE(
		features->get_feature_class() == C_DENSE,
		"Provided features (%d) "
		"has to be of C_DENSE (%d) class!\n",
		features->get_feature_class(), C_DENSE);

	auto matrix = features->as<DenseFeatures<ST>>()->get_feature_matrix();
	if (!inplace)
		matrix = matrix.clone();
	auto feat_matrix = inverse_apply_to_matrix(matrix);
	return std::make_shared<DenseFeatures<ST>>(feat_matrix);
}

template <class ST>
SGMatrix<ST>
DensePreprocessor<ST>::inverse_apply_to_matrix(SGMatrix<ST> matrix)
{
	SG_SNOTIMPLEMENTED;

	return SGMatrix<ST>();
}

template class DensePreprocessor<bool>;
template class DensePreprocessor<char>;
template class DensePreprocessor<int8_t>;
template class DensePreprocessor<uint8_t>;
template class DensePreprocessor<int16_t>;
template class DensePreprocessor<uint16_t>;
template class DensePreprocessor<int32_t>;
template class DensePreprocessor<uint32_t>;
template class DensePreprocessor<int64_t>;
template class DensePreprocessor<uint64_t>;
template class DensePreprocessor<float32_t>;
template class DensePreprocessor<float64_t>;
template class DensePreprocessor<floatmax_t>;
}
