/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _VARIANCEKERNELNORMALIZER_H___
#define _VARIANCEKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief VarianceKernelNormalizer divides by the ``variance''
 *
 * This effectively normalizes the vectors in feature space to variance 1 (see
 * VarianceKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\frac{1}{N}\sum_{i=1}^N k({\bf x}_i, {\bf x}_i) - \sum_{i,j=1}^N, k({\bf x}_i,{\bf x'}_j)/N^2}
 * \f]
 */
class VarianceKernelNormalizer : public KernelNormalizer
{
	public:
		/** default constructor
		 */
		VarianceKernelNormalizer()
			: KernelNormalizer(), meandiff(1.0), sqrt_meandiff(1.0)
		{
			/*SG_ADD(&meandiff, "meandiff", "Scaling constant.", ParameterProperties::HYPER)*/;
			/*SG_ADD(&sqrt_meandiff, "sqrt_meandiff",
					"Square root of scaling constant.", ParameterProperties::HYPER)*/;
		}

		/** default destructor */
		virtual ~VarianceKernelNormalizer()
		{
		}

		/** initialization of the normalizer
         * @param k kernel */
		virtual bool init(Kernel* k)
		{
			ASSERT(k)
			int32_t n=k->get_num_vec_lhs();
			ASSERT(n>0)

			auto old_lhs=k->lhs;
			auto old_rhs=k->rhs;
			k->lhs=old_lhs;
			k->rhs=old_lhs;

			float64_t diag_mean=0;
			float64_t overall_mean=0;
			for (int32_t i=0; i<n; i++)
			{
				diag_mean+=k->compute(i, i);

				for (int32_t j=0; j<n; j++)
					overall_mean+=k->compute(i, j);
			}
			diag_mean/=n;
			overall_mean/=((float64_t) n)*n;

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			meandiff=1.0/(diag_mean-overall_mean);
			sqrt_meandiff = std::sqrt(meandiff);

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
			return value*meandiff;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
		{
			return value*sqrt_meandiff;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
		{
			return value*sqrt_meandiff;
		}

		/** @return object name */
		virtual const char* get_name() const { return "VarianceKernelNormalizer"; }

    protected:
		/** scaling constant */
		float64_t meandiff;
		/** square root of scaling constant */
		float64_t sqrt_meandiff;
};
}
#endif
