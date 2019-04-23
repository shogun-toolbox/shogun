/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Fernando Iglesias, Yuyu Zhang, 
 *          Viktor Gal
 */

#ifndef CUSTOM_MAHALANOBIS_DISTANCE_
#define CUSTOM_MAHALANOBIS_DISTANCE_

#include <shogun/lib/config.h>


#include <shogun/distance/RealDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/**
 * @brief Class CustomMahalanobisDistance used to compute the distance between feature
 * vectors \f$ \vec{x_i} \f$ and \f$ \vec{x_j} \f$ as \f$ (\vec{x_i} - \vec{x_j})^T \mathbf{M}
 * (\vec{x_i} - \vec{x_j}) \f$, given the matrix \f$ \mathbf{M} \f$ which will be referred to as
 * Mahalanobis matrix.
 *
 */
class CustomMahalanobisDistance : public RealDistance
{
	public:
		/** default constructor */
		CustomMahalanobisDistance();

		/** standard constructor
		 *
		 * @param l features of left hand side
		 * @param r features of right hand side
		 * @param m Mahalanobis matrix used to compute distances
		 */
		CustomMahalanobisDistance(std::shared_ptr<Features> l, std::shared_ptr<Features> r, SGMatrix<float64_t> m);

		/** destructor */
		virtual ~CustomMahalanobisDistance();

		/** cleanup distance, here only because it is abstract in Distance. It does nothing */
		virtual void cleanup();

		/** @return name of SGSerializable */
		virtual const char* get_name() const;

		/** get distance type
		 *
		 * @return distance type CUSTOMMAHALANOBIS
		 */
		virtual EDistanceType get_distance_type();

	protected:
		/**
		 * compute distance between feature idx_a in lhs features and feature idx_b
		 * in rhs features
		 *
		 * @param idx_a feature vector in lhs at idx_a
		 * @param idx_b feature vector in rhs at idx_b
		 *
		 * @return distance value
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		/** register parameters */
		void register_params();

	private:
		/** Mahalanobis matrix used to compute distances */
		SGMatrix<float64_t> m_mahalanobis_matrix;

}; /* class CCustomMahalanobisDistance */

} /* namespace shogun */


#endif /* CUSTOM_MAHALANOBIS_DISTANCE_ */
