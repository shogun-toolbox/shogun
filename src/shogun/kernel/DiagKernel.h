/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _DIAGKERNEL_H___
#define _DIAGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
/** @brief The Diagonal Kernel returns a constant for the diagonal and zero
 * otherwise.
 *
 * A kernel that returns zero for all non-diagonal elements and a single
 * constant otherwise, i.e.\f$k({\bf x_i}, {\bf x_j})= \delta_{ij} c\f$
 *
 */
class DiagKernel: public Kernel
{
	public:

		DiagKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param diag diagonal
		 */
		DiagKernel(int32_t size, float64_t diag=1.0);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param diag diagonal
		 */
		DiagKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t diag=1.0);

		~DiagKernel() override;

		EFeatureType get_feature_type() override{ return F_ANY; }

		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		EFeatureClass get_feature_class() override{ return C_ANY; }

		EKernelType get_kernel_type() override { return K_DIAG; }

		const char* get_name() const override { return "DiagKernel"; }

	protected:
		float64_t compute(int32_t idx_a, int32_t idx_b) override
		{
			return idx_a == idx_b ? m_diag : 0;
		}

	protected:
		float64_t m_diag = 1.0;
};
}
#endif /* _DIAGKERNEL_H__ */
