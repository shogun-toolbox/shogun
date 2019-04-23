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

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> EFeatureType DenseDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> EFeatureType DenseDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> EFeatureType DenseDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> EFeatureType DenseDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> EFeatureType DenseDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> EFeatureType DenseDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> EFeatureType DenseDistance<char>::get_feature_type() { return F_CHAR; }

template class DenseDistance<char>;
template class DenseDistance<uint8_t>;
template class DenseDistance<int16_t>;
template class DenseDistance<uint16_t>;
template class DenseDistance<int32_t>;
template class DenseDistance<uint64_t>;
template class DenseDistance<float64_t>;
}
