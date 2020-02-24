#include <shogun/distance/DenseDistance.h>

namespace shogun {

template <class ST> bool DenseDistance<ST>::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Distance::init(l,r);

	ASSERT(l->get_feature_class()==C_DENSE)
	ASSERT(r->get_feature_class()==C_DENSE)
	ASSERT(l->get_feature_type()==this->get_feature_type())
	ASSERT(r->get_feature_type()==this->get_feature_type())


	if ( (std::static_pointer_cast<DenseFeatures<float64_t>>(l))->get_num_features() != (std::static_pointer_cast<DenseFeatures<float64_t>>(r))->get_num_features())
	{
		error("train or test features #dimension mismatch (l:{} vs. r:{})",
				(std::static_pointer_cast<DenseFeatures<float64_t>>(l))->get_num_features(),
				(std::static_pointer_cast<DenseFeatures<float64_t>>(r))->get_num_features());
	}

	return true;
}

/** get feature type the  distance can deal with
 *
 */
template <class ST>
EFeatureType DenseDistance<ST>::get_feature_type()
{
	if constexpr (std::is_same_v<ST, char>)
	{
		return F_CHAR;
	}
	else if constexpr (std::is_same_v<ST, float64_t>)
	{
		return F_DREAL;
	}
	else if constexpr (std::is_same_v<uint8_t, ST>)
	{
		return F_BYTE;
	}
	else if constexpr (std::is_same_v<int16_t, ST>)
	{
		return F_SHORT;
	}
	else if constexpr (std::is_same_v<uint16_t, ST>)
	{
		return F_WORD;
	}
	else if constexpr (std::is_same_v<int32_t, ST>)
	{
		return F_INT;
	}
	else if constexpr (std::is_same_v<uint64_t, ST>)
	{
		return F_ULONG;
	}
	else
	{
		return F_DREAL;
	}
}

template class DenseDistance<char>;
template class DenseDistance<uint8_t>;
template class DenseDistance<int16_t>;
template class DenseDistance<uint16_t>;
template class DenseDistance<int32_t>;
template class DenseDistance<uint64_t>;
template class DenseDistance<float64_t>;
}
