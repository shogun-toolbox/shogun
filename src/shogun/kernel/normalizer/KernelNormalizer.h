/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _KERNELNORMALIZER_H___
#define _KERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

/** normalizer type */
enum ENormalizerType
{
	N_REGULAR = 0,
	N_MULTITASK = 1
};

class Kernel;
/** @brief The class Kernel Normalizer defines a function to post-process kernel values.
 *
 * Formally it defines f(.,.,.)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = f(k({\bf x},{\bf x'}),{\bf x},{\bf x'})
 * \f]
 *
 * examples for f(.,.,.) would be scaling with a constant
 *
 * \f[
 * f(k({\bf x},{\bf x'}), ., .)= \frac{1}{c}\cdot k({\bf x},{\bf x'})
 * \f]
 *
 * as can be found in class CAvgDiagKernelNormalizer, the identity (cf.
 * CIdentityKernelNormalizer), dividing by the Square Root of the product of
 * the diagonal elements which effectively normalizes the vectors in feature
 * space to norm 1 (see CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\sqrt{k({\bf x},{\bf x})k({\bf x'},{\bf x'})}}
 * \f]
 */
class KernelNormalizer : public SGObject
{
	public:

		/** default constructor
		 */
		KernelNormalizer() : SGObject()
		{
			register_params();
			m_type = N_REGULAR;
		}

		/** default destructor */
		virtual ~KernelNormalizer() { }

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		virtual bool init(Kernel* k)=0;

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
			float64_t value, int32_t idx_lhs, int32_t idx_rhs) const=0;

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const=0;

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const=0;

		/** getter for normalizer type
		 */
		ENormalizerType get_normalizer_type() const noexcept
		{
			return m_type;
		}

	protected:
		/** register the parameters
		 */
		virtual void register_params()
		{
			SG_ADD_OPTIONS(
			    (machine_int_t*)&m_type, "m_type", "Normalizer type.",
			    ParameterProperties::NONE, SG_OPTIONS(N_REGULAR, N_MULTITASK));
		}

		/** setter for normalizer type
		 *  @param type type of normalizer
		 */
		void set_normalizer_type(ENormalizerType type)
		{
			m_type = type;
		}

		/** normalizer type */
		ENormalizerType m_type;
};
}
#endif
