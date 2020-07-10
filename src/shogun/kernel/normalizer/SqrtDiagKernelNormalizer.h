/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _SQRTDIAGKERNELNORMALIZER_H___
#define _SQRTDIAGKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
/** @brief SqrtDiagKernelNormalizer divides by the Square Root of the product of
 * the diagonal elements.
 *
 * This effectively normalizes the vectors in feature space to norm 1 (see
 * CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\sqrt{k({\bf x},{\bf x})k({\bf x'},{\bf x'})}}
 * \f]
 */
class SqrtDiagKernelNormalizer : public KernelNormalizer
{
	public:
		SqrtDiagKernelNormalizer(): KernelNormalizer()
		{
			SG_ADD(&sqrtdiag_lhs, "sqrtdiag_lhs",
				"sqrt(K(x,x)) for left hand side examples.")

			SG_ADD(&sqrtdiag_rhs, "sqrtdiag_rhs",
				"sqrt(K(x,x)) for right hand side examples.")

			SG_ADD(&use_optimized_diagonal_computation,
					"use_optimized_diagonal_computation",
					"flat if optimized diagonal computation is used");
		}

		/** default constructor
		 * @param use_opt_diag - some kernels support faster diagonal compuation
		 * via compute_diag(idx), this flag enables this
		 */
		SqrtDiagKernelNormalizer(bool use_opt_diag): SqrtDiagKernelNormalizer()
		{
			use_optimized_diagonal_computation = use_opt_diag;
		}

		/** default destructor */
		~SqrtDiagKernelNormalizer() override = default;

		/** initialization of the normalizer
         * @param k kernel */
		bool init(Kernel* k) override
		{
			ASSERT(k)
			const auto& num_sqrtdiag_lhs=k->get_num_vec_lhs();
			const auto& num_sqrtdiag_rhs=k->get_num_vec_rhs();
			ASSERT(num_sqrtdiag_lhs>0)
			ASSERT(num_sqrtdiag_rhs>0)

			auto old_lhs=k->lhs;
			auto old_rhs=k->rhs;

			k->lhs=old_lhs;
			k->rhs=old_lhs;
			sqrtdiag_lhs = alloc_and_compute_diag(k, num_sqrtdiag_lhs);

			k->lhs=old_rhs;
			k->rhs=old_rhs;
			sqrtdiag_rhs = alloc_and_compute_diag(k, num_sqrtdiag_rhs);

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
			float64_t sqrt_both=sqrtdiag_lhs[idx_lhs]*sqrtdiag_rhs[idx_rhs];
			return value/sqrt_both;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const override
		{
			return value/sqrtdiag_lhs[idx_lhs];
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const override
		{
			return value/sqrtdiag_rhs[idx_rhs];
		}

		/** @return object name */
		const char* get_name() const override { return "SqrtDiagKernelNormalizer"; }

    protected:
		/**
		 * alloc and compute the vector containing the square root of the
		 * diagonal elements of this kernel.
		 */
		SGVector<float64_t> alloc_and_compute_diag(Kernel* k, int32_t num) const
		{
			SGVector<float64_t> v(num);
			std::function<float64_t(const int32_t)> func;

			if (auto* cwk = dynamic_cast<CommWordStringKernel*>(k))
			{
				if (use_optimized_diagonal_computation)
					func = [&cwk] (const int32_t i) {return std::sqrt(cwk->compute_diag(i));};
				else
					func = [&cwk] (const int32_t i) { return std::sqrt(cwk->compute_helper(i,i, true)); };
			}
			else 
				func = [&k] (const int32_t i) { return std::sqrt(k->compute(i,i));};

			for (int32_t i=0; i<num; i++)
			{
				v[i] = func(i);
				if (v[i] == 0.0)
					v[i] = std::numeric_limits<float64_t>::min();
			}

			return v;
		}

		/** sqrt diagonal left-hand side */
		SGVector<float64_t> sqrtdiag_lhs;

		/** sqrt diagonal right-hand side */
		SGVector<float64_t> sqrtdiag_rhs;

		/** f optimized diagonal computation is used */
		bool use_optimized_diagonal_computation = false;
};
}
#endif
