/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SPARSEDISTANCE_H___
#define _SPARSEDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/distance/Distance.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
/** @brief template class SparseDistance */
template <class ST> class CSparseDistance : public CDistance
{
	public:
		/** default constructor */
		CSparseDistance() : CDistance() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CDistance::init(l,r);

			ASSERT(l->get_feature_class()==C_SPARSE)
			ASSERT(r->get_feature_class()==C_SPARSE)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())

			if (((CSparseFeatures<ST>*) lhs)->get_num_features() != ((CSparseFeatures<ST>*) rhs)->get_num_features() )
			{
				SG_ERROR("train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSparseFeatures<ST>*) lhs)->get_num_features(),((CSparseFeatures<ST>*)rhs)->get_num_features());
			}
			return true;
		}

		/** get feature class the distance can deal with
		 *
		 * @return feature class SPARSE
		 */
		virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		/** get feature type the distance can deal with
		 *
		 * @return template-specific feature type
		 */
		virtual EFeatureType get_feature_type();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "SparseDistance"; }

		/** cleanup distance
		 *
		 * abstract base method
		 */
		virtual void cleanup()=0;

		/** get distance type we are
		 *
		 * abstrace base method
		 *
		 * @return distance type
		 */
		virtual EDistanceType get_distance_type()=0;
};

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CSparseDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CSparseDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CSparseDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CSparseDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CSparseDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CSparseDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType CSparseDistance<char>::get_feature_type() { return F_CHAR; }
} // namespace shogun
#endif
