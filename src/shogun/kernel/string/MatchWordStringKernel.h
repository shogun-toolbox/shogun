/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _MATCHWORDSTRINGKERNEL_H___
#define _MATCHWORDSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
/** @brief The class MatchWordStringKernel computes a variant of the polynomial
 * kernel on strings of same length converted to a word alphabet.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= \sum_{i=0}^L I(x_i=x'_i)+c)^d
 * \f]
 *
 * where I is the indicator function which evaluates to 1 if its argument is
 * true and to 0 otherwise.
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class MatchWordStringKernel: public StringKernel<uint16_t>
{
	public:
		/** default constructor  */
		MatchWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param d degree
		 */
		MatchWordStringKernel(int32_t size, int32_t d);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		MatchWordStringKernel(const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r, int32_t degree);

		virtual ~MatchWordStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type LINEAR
		 */
		virtual EKernelType get_kernel_type() { return K_MATCHWORD; }

		/** return the kernel's name
		 *
		 * @return name MatchWordString
		 */
		virtual const char* get_name() const { return "MatchWordStringKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** degree */
		int32_t degree;
};
}
#endif /* _MATCHWORDSTRINGKERNEL_H__ */
