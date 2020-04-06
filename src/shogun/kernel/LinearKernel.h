/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser
 */

#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{
	class KernelMachine;
	class DotFeatures;

/** @brief Computes the standard linear kernel on DotFeatures.
 *
 * Formally, it computes
 *
 * \f[
 * k({\bf x},{\bf x'})= {\bf x}\cdot {\bf x'}
 * \f]
 */
class LinearKernel: public DotKernel
{
	public:
		/** constructor
		 */
		LinearKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		LinearKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r);

		~LinearKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type LINEAR
		 */
		EKernelType get_kernel_type() override { return K_LINEAR; }

		/** return the kernel's name
		 *
		 * @return name Lineaer
		 */
		const char* get_name() const override { return "LinearKernel"; }

		/** optimizable kernel, i.e. precompute normal vector and as
		 * phi(x) = x do scalar product in input space
		 *
		 * @param num_suppvec number of support vectors
		 * @param sv_idx support vector index
		 * @param alphas alphas
		 * @return if optimization was successful
		 */
		bool init_optimization(
			int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas) override;

		/** init optimization
		 * @param km
		 */
		virtual bool init_optimization(std::shared_ptr<KernelMachine> km);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		bool delete_optimization() override;

		/** compute optimized
		*
		* @param idx index to compute
		* @return optimized value at given index
		*/
		float64_t compute_optimized(int32_t idx) override;

		void clear_normal() override
		{
			normal = SGVector<float64_t>((std::static_pointer_cast<DotFeatures>(lhs))->get_dim_feature_space());
			normal.zero();
			set_is_initialized(false);
		}

		/** add to normal vector
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		void add_to_normal(int32_t idx, float64_t weight) override;

		/** get normal vector
		 *
		 * @return normal vector
		 */
		SGVector<float64_t> get_w() const
		{
			ASSERT(lhs)
			return normal;
		}

		/** set normal vector
		 *
		 * @param w new normal
		 */
		void set_w(SGVector<float64_t> w)
		{
			ASSERT(lhs && w.size()==(std::static_pointer_cast<DotFeatures>(lhs))->get_dim_feature_space())
			this->normal = w;
		}

	protected:
		/** normal vector (used in case of optimized kernel) */
		SGVector<float64_t> normal;
};
}
#endif /* _LINEARKERNEL_H__ */
