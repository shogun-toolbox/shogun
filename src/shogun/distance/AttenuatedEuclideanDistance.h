/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _ATTENUATEDEuclideanDISTANCE_H__
#define _ATTENUATEDEuclideanDISTANCE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/RealDistance.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief class AttenuatedEuclideanDistance
 *
 * The adaptation of the familiar Euclidean Distance, to be used in
 * ternary ECOC designs. This adaptation computes the Euclidean distance
 * between two vectors ignoring those positions of either of the vectors
 * valued as 0. Note that this might make sense only in the Decoding
 *  step of the ECOC framework, since the 0 value denotes that a certain category is
 *  ignored.
 *
 * This distance was proposed by
 * S. Escalera, O. Pujol, P.Radeva in On the decoding process in ternary error-correcting output codes,
 * Transactions in Pattern Analysis and Machine Intelligence 99 (1).
 *
 * \f[\displaystyle
 *  d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{n}|x_i||x'_i|{\bf x_i}-{\bf x'_i}|^2}
 * \f]
 *
 */
class AttenuatedEuclideanDistance: public RealDistance
{
	public:
		/** default constructor */
		AttenuatedEuclideanDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		AttenuatedEuclideanDistance(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~AttenuatedEuclideanDistance() override;

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
		 * @return distance type Euclidean
		 */
		EDistanceType get_distance_type() override { return D_ATTENUATEDEUCLIDEAN; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		EFeatureType get_feature_type() override { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name Euclidean
		 */
		const char* get_name() const override { return "AttenuatedEuclideanDistance"; }

		/** disable application of sqrt on matrix computation
		 * the matrix can then also be named norm squared
		 *
		 * @return if application of sqrt is disabled
		 */
		virtual bool get_disable_sqrt() { return disable_sqrt; };

		/** disable application of sqrt on matrix computation
		 * the matrix can then also be named norm squared
		 *
		 * @param state new disable_sqrt
		 */
		virtual void set_disable_sqrt(bool state) { disable_sqrt=state; };

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	private:
		void init();

	protected:
		/** if application of sqrt on matrix computation is disabled */
		bool disable_sqrt;
};

} // namespace shogun
#endif /* _ATTENUATEDEuclideanDISTANCE_H__ */
