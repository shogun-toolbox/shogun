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
	template <class T> class SparseFeatures;
/** @brief class SparseEucldeanDistance */
class SparseEuclideanDistance: public SparseDistance<float64_t>
{
	public:
		/** default constructor */
		SparseEuclideanDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		SparseEuclideanDistance(
			const std::shared_ptr<SparseFeatures<float64_t>>& l, const std::shared_ptr<SparseFeatures<float64_t>>& r);
		~SparseEuclideanDistance() override;

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup distance */
		void cleanup() override;

		/** get distance type we are
		 *
		 * @return distance type SPARSEEUCLIDEAN
		 */
		EDistanceType get_distance_type() override { return D_SPARSEEUCLIDEAN; }

		/** get supported feature type
		 *
		 * @return feature type DREAL
		 */
		EFeatureType get_feature_type() override { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name SparseEuclidean
		 */
		const char* get_name() const override { return "SparseEuclideanDistance"; }

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
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
