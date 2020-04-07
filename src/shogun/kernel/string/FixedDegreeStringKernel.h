/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _FIXEDDEGREESTRINGKERNEL_H___
#define _FIXEDDEGREESTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The FixedDegree String kernel takes as input two strings of same size
 * and counts the number of matches of length d.
 *
 * \f[
 *     k({\bf x}, {\bf x'}) = \sum_{i=0}^{l-d} I({\bf x}_{i,i+1,\dots,i+d-1} = {\bf x'}_{i,i+1,\dots,i+d-1})
 * \f]
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class FixedDegreeStringKernel: public StringKernel<char>
{
	void init();

	public:
		/** default constructor  */
		FixedDegreeStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree the degree
		 */
		FixedDegreeStringKernel(int32_t size, int32_t degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree the degree
		 */
		FixedDegreeStringKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
			int32_t degree);

		~FixedDegreeStringKernel() override;

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
		 * @return kernel type FIXEDDEGREE
		 */
		EKernelType get_kernel_type() override
		{
			return K_FIXEDDEGREE;
		}

		/** return the kernel's name
		 *
		 * @return name FixedDegree
		 */
		const char* get_name() const override{ return "FixedDegreeStringKernel"; }

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
		/** the degree */
		int32_t degree;
};
}
#endif /* _FIXEDDEGREESTRINGKERNEL_H___ */
