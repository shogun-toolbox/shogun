/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Yuyu Zhang, Viktor Gal,
 *          Evan Shelhamer, Sergey Lisitsyn
 */

#ifndef _MAHALANOBISDISTANCE_H__
#define _MAHALANOBISDISTANCE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/RealDistance.h>

namespace shogun
{
	/** @brief class MahalanobisDistance
	 *
	 * The Mahalanobis distance for real valued features computes the distance
	 * between a feature vector and a distribution of features characterized by
	 * its
	 * mean and covariance.
	 *
	 * \f[\displaystyle
	 *  D = \sqrt{ (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)  }
	 * \f]
	 *
	 * The Mahalanobis Squared distance does not take the square root:
	 *
	 * \f[\displaystyle
	 *  D = (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)
	 * \f]
	 *
	 * If use_mean is set to false (which it is by default) the distance is
	 * computed
	 * as
	 *
	 * \f[\displaystyle
	 *  D = \sqrt{ (x_i - x_i')^T \Sigma^{-1} (x_i - x_i')  }
	 * \f]
	 *
	 * i.e., instead of the mean as reference two vector \f$x_i\f$ and
	 * \f$x_i'\f$
	 * are compared.
	 *
	 * @see <a href="http://en.wikipedia.org/wiki/Mahalanobis_distance">
	 * Wikipedia: Mahalanobis Distance</a>
	 */
	class MahalanobisDistance : public RealDistance
	{
	public:
		/** default constructor */
		MahalanobisDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		MahalanobisDistance(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~MahalanobisDistance() override;

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
		 * @return distance type MAHALANOBIS
		 */
		EDistanceType get_distance_type() override { return D_MAHALANOBIS; }

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		EFeatureType get_feature_type() override { return F_DREAL; }

		/** get name of the distance
		 *
		 * @return name Mahalanobis
		 */
		const char* get_name() const override { return "MahalanobisDistance"; }

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

		/** whether the distance is computed between the mean and a vector of rhs
		 * or between lhs and rhs
		 *
		 * @return if the mean of lhs is used to obtain the distance
		 */
		virtual bool get_use_mean() { return use_mean; };

		/** whether the distance is computed between the mean and a vector of rhs
		 * or between lhs and rhs
		 *
		 * @param state new use_mean
		 */
		virtual void set_use_mean(bool state) { use_mean=state; };

	protected:
		/// compute Mahalanobis distance between a feature vector of lhs
		/// to a feature vector of rhs
		/// if use_mean then idx_a is not used and the distance
		/// computed is between a feature vector of rhs and the
		/// distribution lhs
		///
		/// @param idx_a index of the feature vector in lhs
		/// @param idx_b index of the feature vector in rhs
		/// @return value of the Mahalanobis distance
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	private:
		void init();

	protected:
		/** if application of sqrt on matrix computation is disabled */
		bool disable_sqrt;

		/** whether the features lhs and rhs have exactly the same values */
		bool use_mean;

		/** vector mean of the lhs feature vectors */
		SGVector<float64_t> mean;

		/** LDLT decomposition of the covariance matrix of lhs feature vectors
		 */
		SGMatrix<float64_t> chol_cov_L;
		SGVector<float64_t> chol_cov_d;
		SGVector<index_t> chol_cov_p;
};

} // namespace shogun
#endif /* _MAHALANOBISDISTANCE_H__ */
