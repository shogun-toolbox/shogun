#include <shogun/distance/DenseDistance.h>

namespace shogun {

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> EFeatureType CDenseDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> EFeatureType CDenseDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> EFeatureType CDenseDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> EFeatureType CDenseDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> EFeatureType CDenseDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> EFeatureType CDenseDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> EFeatureType CDenseDistance<char>::get_feature_type() { return F_CHAR; }

template class CDenseDistance<char>;
template class CDenseDistance<uint8_t>;
template class CDenseDistance<int16_t>;
template class CDenseDistance<uint16_t>;
template class CDenseDistance<int32_t>;
template class CDenseDistance<uint64_t>;
template class CDenseDistance<float64_t>;
}
