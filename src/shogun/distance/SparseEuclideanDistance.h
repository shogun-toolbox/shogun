/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SPARSEEUCLIDEANDISTANCE_H__
#define _SPARSEEUCLIDEANDISTANCE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/SparseDistance.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
	template <class T> class CSparseFeatures;
/** @brief class SparseEucldeanDistance */
class CSparseEuclideanDistance: public CSparseDistance<float64_t>
{
	public:
		/** default constructor */
		CSparseEuclideanDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CSparseEuclideanDistance(
			CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r);
		virtual ~CSparseEuclideanDistance();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** cleanup distance */
		virtual void cleanup();

		/** get distance type we are
		 *
		 * @return distance type SPARSEEUCLIDEAN
		 */
		virtual EDistanceType get_distance_type() { return D_SPARSEEUCLIDEAN; }

		/** get supported feature type
		 *
		 * @return feature type DREAL
		 */
		virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name SparseEuclidean
		 */
		virtual const char* get_name() const { return "SparseEuclideanDistance"; }

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
		/*    compute_kernel*/

	private:
		void init();

	protected:
		/** squared left-hand side */
		float64_t* sq_lhs;
		/** squared right-hand side */
		float64_t* sq_rhs;

};

} // namespace shogun
#endif /* _SPARSEEUCLIDEANDISTANCE_H__ */
