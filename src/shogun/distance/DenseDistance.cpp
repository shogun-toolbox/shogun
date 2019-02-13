#include <shogun/distance/DenseDistance.h>

namespace shogun {

template <class ST> bool CDenseDistance<ST>::init(CFeatures* l, CFeatures* r)
{
	CDistance::init(l,r);

	ASSERT(l->get_feature_class()==C_DENSE)
	ASSERT(r->get_feature_class()==C_DENSE)
	ASSERT(l->get_feature_type()==this->get_feature_type())
	ASSERT(r->get_feature_type()==this->get_feature_type())


	if ( ((CDenseFeatures<ST>*) l)->get_num_features() != ((CDenseFeatures<ST>*) r)->get_num_features() )
	{
		SG_ERROR("train or test features #dimension mismatch (l:%d vs. r:%d)\n",
				((CDenseFeatures<ST>*) l)->get_num_features(),((CDenseFeatures<ST>*) r)->get_num_features());
	}

	return true;
}

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> SHOGUN_EXPORT EFeatureType CDenseDistance<char>::get_feature_type() { return F_CHAR; }

template class SHOGUN_EXPORT CDenseDistance<char>;
template class SHOGUN_EXPORT CDenseDistance<uint8_t>;
template class SHOGUN_EXPORT CDenseDistance<int16_t>;
template class SHOGUN_EXPORT CDenseDistance<uint16_t>;
template class SHOGUN_EXPORT CDenseDistance<int32_t>;
template class SHOGUN_EXPORT CDenseDistance<uint64_t>;
template class SHOGUN_EXPORT CDenseDistance<float64_t>;
}
