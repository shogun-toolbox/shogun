/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _ZEROMEANCENTERKERNELNORMALIZER_H___
#define _ZEROMEANCENTERKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief ZeroMeanCenterKernelNormalizer centers the kernel in feature space
 *
 * After centering, each feature must have zero mean.  The centered kernel
 * matrix can be expressed in terms of the non-centered version.
 *
 * Denoting the mapping from input space to feature space by \f$\phi:\mathcal{X}\rightarrow\mathcal{F}\f$, the centered square kernel matrix \f$K_c\f$ (with  dimensionality \f$ M \f$)
 *
 * can be expressed in terms of the original matrix \f$K\f$ as follows:
 *
 * \f{eqnarray*}
 * k({\bf x}_i,{\bf x}_j)_c & = & \left(\phi({\bf x}_i) - \frac{1}{m} \sum_{p=1}^M \phi({\bf x}_p)\right) \cdot \left(\phi({\bf x}_j) - \frac{1}{M} \sum_{q=1}^M \phi({\bf x}_q)\right)  \\
 *          & = & K_{ij} - \frac{1}{M} \sum_{p=1}^M K_{pj} - \frac{1}{M} \sum_{q=1}^M K_{iq} + \frac{1}{M^2} \sum_{p=1}^M \sum_{q=1}^M K_{pq} \\
 *          & = & (K - 1_M K - K 1_M + 1_M K 1_M)_{ij}
 * \f}
 *
 *
 * Additionally, let  \f$ K^{t} \f$  be the \f$ L \times M \f$ test matrix describing the similarity between a \f$ L \f$ test instances with \f$M\f$ training instances
 *
 * (defined by a \f$ M x M \f$ kernel matrix \f$ K\f$), the centered testing set kernel matrix is given by
 * \f[
 * K_{c}^t  =  (K - 1'_M K - K^{t} 1_M + 1'_M K 1_M)
 * \f]
 */
class ZeroMeanCenterKernelNormalizer : public KernelNormalizer
{
	public:
		/** default constructor
		*/
		ZeroMeanCenterKernelNormalizer()
		{
			SG_ADD(&ktrain_row_means, "num_ktrain", "Train row means.", 
				ParameterProperties::MODEL)
			SG_ADD(&ktrain_row_means, "num_ktest", "Test row means.",
				ParameterProperties::MODEL)
		}

		/** default destructor */
		virtual ~ZeroMeanCenterKernelNormalizer() = default;

		/** initialization of the normalizer
		 * @param k kernel */
		bool init(Kernel* k) override
		{
			ASSERT(k)
			const auto& num_lhs=k->get_num_vec_lhs();
			const auto& num_rhs=k->get_num_vec_rhs();
			ASSERT(num_lhs>0)
			ASSERT(num_rhs>0)

			auto old_lhs=k->lhs;
			auto old_rhs=k->rhs;

			/* compute mean for each row of the train matrix*/
			k->lhs=old_lhs;
			k->rhs=old_lhs;

			ktrain_row_means = alloc_and_compute_row_means(k, num_lhs, num_lhs);

			/* compute mean for each row of the test matrix */
			k->lhs=old_lhs;
			k->rhs=old_rhs;

			ktest_row_means = alloc_and_compute_row_means(k, num_lhs, num_rhs);

			/* compute train kernel matrix mean */
			ktrain_mean=0;
			for (int32_t i=0;i<num_lhs;i++)
				ktrain_mean += ktrain_row_means[i] / num_lhs;

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			return true;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
				float64_t value, int32_t idx_lhs, int32_t idx_rhs) const
		{
			value += (-ktrain_row_means[idx_lhs] - ktest_row_means[idx_rhs] + ktrain_mean);
			return value;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
		{
			error("normalize_lhs not implemented");
			return 0;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
		{
			error("normalize_rhs not implemented");
			return 0;
		}

		/**
		 * alloc and compute the vector containing the row margins of all rows
		 * for a kernel matrix.
		 */
		SGVector<float64_t> alloc_and_compute_row_means(Kernel* k, int32_t num_lhs, int32_t num_rhs) const
		{
			SGVector<float64_t> v(num_rhs);

			for (int32_t i=0; i<num_rhs; i++)
			{
				for (int32_t j=0; j<num_lhs; j++)
					v[i] += ( k->compute(j,i)/num_lhs );
			}
			return v;
		}

		/** @return object name */
		virtual const char* get_name() const { return "ZeroMeanCenterKernelNormalizer"; }

	protected:
		/** train row means */
		SGVector<float64_t> ktrain_row_means;

		SGVector<float64_t> ktest_row_means;

		/** train mean */
		float64_t ktrain_mean;
};
}
#endif
