/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _DICEKERNELNORMALIZER_H___
#define _DICEKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
/** @brief DiceKernelNormalizer performs kernel normalization inspired by the Dice
 * coefficient (see http://en.wikipedia.org/wiki/Dice's_coefficient)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{2k({\bf x},{\bf x'})}{k({\bf x},{\bf x})+k({\bf x'},{\bf x'}}
 * \f]
 */
class DiceKernelNormalizer : public KernelNormalizer
{
	public:
		DiceKernelNormalizer(): KernelNormalizer()
		{
			SG_ADD(&diag_rhs, "diag_rhs", "K(x,x) for right hand side examples.",
				   ParameterProperties::MODEL)

			SG_ADD(&diag_lhs, "diag_lhs", "K(x,x) for left hand side examples.",
				   ParameterProperties::MODEL)

			SG_ADD(&use_optimized_diagonal_computation,
				   "use_optimized_diagonal_computation",
				   "flat if optimized diagonal computation is used");
		}

		/** constructor
		 * @param use_opt_diag - some kernels support faster diagonal compuation
		 * via compute_diag(idx), this flag enables this
		 */
		DiceKernelNormalizer(bool use_opt_diag): KernelNormalizer()
		{
			use_optimized_diagonal_computation = use_opt_diag;
		}

		/** default destructor */
		~DiceKernelNormalizer() override = default;

		/** initialization of the normalizer
         * @param k kernel */
		bool init(Kernel* k) override
		{
			ASSERT(k)
			const auto& num_diag_lhs=k->get_num_vec_lhs();
			const auto& num_diag_rhs=k->get_num_vec_rhs();
			ASSERT(num_diag_lhs>0)
			ASSERT(num_diag_rhs>0)

			auto old_lhs=k->lhs;
			auto old_rhs=k->rhs;

			k->lhs=old_lhs;
			k->rhs=old_lhs;
			diag_lhs=alloc_and_compute_diag(k, num_diag_lhs);

			k->lhs=old_rhs;
			k->rhs=old_rhs;
			diag_rhs=alloc_and_compute_diag(k, num_diag_rhs);

			k->lhs=old_lhs;
			k->rhs=old_rhs;

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
			float64_t diag_sum=diag_lhs[idx_lhs]*diag_rhs[idx_rhs];
			return 2*value/diag_sum;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const override
		{
			error("linadd not supported with Dice normalization.");
			return 0;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const override
		{
			error("linadd not supported with Dice normalization.");
			return 0;
		}

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		const char* get_name() const override {
			return "DiceKernelNormalizer"; }

    protected:
		/**
		 * alloc and compute the vector containing the square root of the
		 * diagonal elements of this kernel.
		 */
		SGVector<float64_t> alloc_and_compute_diag(Kernel* k, int32_t num) const
		{
			SGVector<float64_t> v(num);
			std::function<float64_t(const int32_t)> func;
			
			if (auto* cwsk = dynamic_cast<CommWordStringKernel*>(k))
			{
				if (use_optimized_diagonal_computation)
					func = [&cwsk](const int32_t i) { return cwsk->compute_diag(i); };
				else
					func = [&cwsk](const int32_t i) { return cwsk->compute_helper(i,i, true);};
			}
			else
				func = [&k](const int32_t i) { return k->compute(i,i);};

			for (int32_t i=0; i<num; i++)
			{
				v[i] = func(i);
				if (v[i] == 0.0)
					v[i] = std::numeric_limits<float64_t>::min();
			}

			return v;
		}

		/** diagonal left-hand side */
		SGVector<float64_t> diag_lhs;

		/** diagonal right-hand side */
		SGVector<float64_t> diag_rhs;

		/** flat if optimized diagonal computation is used */
		bool use_optimized_diagonal_computation = false;
};
}
#endif
