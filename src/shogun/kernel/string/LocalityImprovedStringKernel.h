/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _LOCALITYIMPROVEDSTRINGKERNEL_H___
#define _LOCALITYIMPROVEDSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The LocalityImprovedString kernel is inspired by the polynomial kernel.
 * Comparing neighboring characters it puts emphasize on local features.
 *
 * It can be defined as
 * \f[
 * K({\bf x},{\bf x'})=\left(\sum_{i=0}^{T-1}\left(\sum_{j=-l}^{+l}w_jI_{i+j}({\bf x},{\bf x'})\right)^{d_1}\right)^{d_2},
 * \f]
 * where
 * \f$ I_i({\bf x},{\bf x'})=1\f$ if \f$x_i=x'_i\f$ and 0 otherwise.
 */
class LocalityImprovedStringKernel: public StringKernel<char>
{
	public:
		/** default constructor  */
		LocalityImprovedStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param length length
		 * @param inner_degree inner degree
		 * @param outer_degree outer degree
		 */
		LocalityImprovedStringKernel(int32_t size, int32_t length,
			int32_t inner_degree, int32_t outer_degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param length length
		 * @param inner_degree inner degree
		 * @param outer_degree outer degree
		 */
		LocalityImprovedStringKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
			int32_t length, int32_t inner_degree, int32_t outer_degree);

		virtual ~LocalityImprovedStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type LOCALITYIMPROVED
		 */
		virtual EKernelType get_kernel_type() { return K_LOCALITYIMPROVED; }

		/** return the kernel's name
		 *
		 * @return name LocalityImprovedStringKernel
		 */
		virtual const char* get_name() const { return "LocalityImprovedStringKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** length */
		int32_t length;
		/** inner degree */
		int32_t inner_degree;
		/** outer degree */
		int32_t outer_degree;
};
}
#endif /* _LOCALITYIMPROVEDSTRINGKERNEL_H__ */
