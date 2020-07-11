/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _RIDGEKERNELNORMALIZER_H___
#define _RIDGEKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief Normalize the kernel by adding a constant term to its diagonal.
 * This aids kernels to become positive definite (even though they are
 * not - often caused by numerical problems).
 *
 * Formally,
 *
 * \f[
 * k'(x,x')= \frac{k(x,x')}+ R\cdot {\bf E}
 * \f]
 *
 * where E is a matrix with ones on the diagonal and R is the scalar
 * ridge term. The ridge term R is computed as \f$R=r\dot c\f$.
 *
 * Typically,
 *
 * - r=1e-10 and c=0.0 will add mean(diag(K))*1e-10 to the diagonal
 * - r=0.1 and c=1 will add 0.1 to the diagonal
 *
 *
 * In case c <= 0, c is compute as the mean of the kernel diagonal
 * \f[
 * \mbox{c} = \frac{1}{N}\sum_{i=1}^N k(x_i,x_i)
 * \f]
 *
 */
class RidgeKernelNormalizer : public KernelNormalizer
{
	public:

		RidgeKernelNormalizer()
			: KernelNormalizer()
		{
			SG_ADD(&scale, "scale", "Scale quotient by which kernel is scaled.",
			    ParameterProperties::HYPER);
			SG_ADD(&ridge, "ridge", "Ridge added to diagonal.", ParameterProperties::HYPER);
		}

		/** constructor
		 *
		 * @param r ridge parameter
		 * @param c scale parameter, if <= 0 scaling will be computed
		 * from the avg of the kernel diagonal elements
		 *
		 * the scalar r*c will be added to the kernel diagonal, typical use cases:
		 * - r=1e-10 and c=0.0 will add mean(diag(K))*1e-10 to the diagonal
		 * - r=0.1 and c=1 will add 0.1 to the diagonal
		 */
		RidgeKernelNormalizer(float64_t r, float64_t c)
			: RidgeKernelNormalizer()
		{
			scale=c;
			ridge=r;
		}

		/** default destructor */
		~RidgeKernelNormalizer() override = default;

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		bool init(Kernel* k) override
		{
			if (scale<=0)
			{
				ASSERT(k)
				const auto& num=k->get_num_vec_lhs();
				ASSERT(num>0)

				auto old_lhs=k->lhs;
				auto old_rhs=k->rhs;
				k->lhs=old_lhs;
				k->rhs=old_lhs;

				float64_t sum=0;
				for (int32_t i=0; i<num; i++)
					sum+=k->compute(i, i);

				scale=sum/num;
				
				k->lhs=old_lhs;
				k->rhs=old_rhs;
			}

			ridge*=scale;
			return true;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		float64_t normalize(
			float64_t value, int32_t idx_lhs, int32_t idx_rhs) const override
		{
			if (idx_lhs==idx_rhs)
				return value+ridge;
			else
				return value;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const override
		{
			error("linadd not supported with Ridge normalization.");
			return 0;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const override
		{
			error("linadd not supported with Ridge normalization.");
			return 0;
		}

		/** @return object name */
		const char* get_name() const override { return "RidgeKernelNormalizer"; }

	protected:
		/// the constant ridge to be added to the kernel diagonal
		float64_t ridge = 1e-10;
		/// scaling parameter (avg of diagonal)
		float64_t scale = 0.0;
};
}
#endif // _RIDGEKERNELNORMALIZER_H___
