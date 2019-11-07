/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _AVGDIAGKERNELNORMALIZER_H___
#define _AVGDIAGKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
namespace shogun
{
/** @brief Normalize the kernel by either a constant or the average value of the
 * diagonal elements (depending on argument c of the constructor).
 *
 * In case c <= 0 compute scale as
* \f[
* \mbox{scale} = \frac{1}{N}\sum_{i=1}^N k(x_i,x_i)
* \f]
*
* otherwise use scale=c and normalize the kernel via
*
* \f[
* k'(x,x')= \frac{k(x,x')}{scale}
* \f]
*/
class AvgDiagKernelNormalizer : public KernelNormalizer
{
	public:
		/** constructor
		 *
		 * @param c scale parameter, if <= 0 scaling will be computed from the
		 * avg of the kernel diagonal elements
		 */
		AvgDiagKernelNormalizer(float64_t c=0.0) : KernelNormalizer()
		{
			scale=c;

			/*SG_ADD(&scale, "scale", "Scale quotient by which kernel is scaled.",
			    ParameterProperties::HYPER)*/;
		}

		/** default destructor */
		virtual ~AvgDiagKernelNormalizer()
		{
		}

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		virtual bool init(Kernel* k)
		{
			if (scale<=0)
			{
				ASSERT(k)
				int32_t num=k->get_num_vec_lhs();
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
			return value/scale;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
		{
			return value/sqrt(scale);
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
		{
			return value/sqrt(scale);
		}

		/** @return object name */
		virtual const char* get_name() const { return "AvgDiagKernelNormalizer"; }

	protected:
		/// the constant scaling factor (avg of diagonal or user given const)
		float64_t scale;
};
}
#endif
