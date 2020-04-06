/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser, Viktor Gal
 */

#ifndef _LINEARSTRINGKERNEL_H___
#define _LINEARSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief Computes the standard linear kernel on dense char valued features.
 *
 * Formally, it computes
 *
 * \f[
 * k({\bf x},{\bf x'})= \frac{1}{scale}{\bf x}\cdot {\bf x'}
 * \f]
 *
 * Note: Basically the same as LinearByteKernel but on signed chars.
 */
class LinearStringKernel: public StringKernel<char>
{
	public:
		/** constructor
		 */
		LinearStringKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		LinearStringKernel(const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r);

		~LinearStringKernel() override;

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
		EKernelType get_kernel_type() override
		{
			return K_LINEAR;
		}

		/** return the kernel's name
		 *
		 * @return name Linear
		 */
		const char* get_name() const override { return "LinearStringKernel"; }

		/** optimizable kernel, i.e. precompute normal vector and as phi(x) = x
		 * do scalar product in input space
		 *
		 * @param num_suppvec number of support vectors
		 * @param sv_idx support vector index
		 * @param alphas alphas
		 * @return if optimization was successful
		 */
		bool init_optimization(
			int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas) override;

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

		/** clear normal */
		void clear_normal() override;

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		void add_to_normal(int32_t idx, float64_t weight) override;

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	protected:
		/** normal vector (used in case of optimized kernel) */
		SGVector<float64_t> m_normal;
};
}
#endif /* _LINEARSTRINGKERNEL_H___ */
