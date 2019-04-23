/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang
 */

#ifndef _POLYKERNEL_H___
#define _POLYKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
	class DotFeatures;

/** @brief Computes the standard polynomial kernel on DotFeatures
 *
 * Formally, it computes
 *
 * \f[
 * k({\bf x},{\bf x'})= (gamma * {\bf x}\cdot {\bf x'}+c)^d
 * \f]
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class PolyKernel: public DotKernel
{
	public:
		/** default constructor  */
		PolyKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param d degree
		 * @param c trade-off parameter
		 * @param size cache size
		 */
		PolyKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r,
			int32_t d, float64_t c, float64_t gamma, int32_t size=10);

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param c trade-off parameter
		 * @param gamma scaler for the dot product
		 */
		PolyKernel(int32_t size, int32_t degree, float64_t c=1.0, float64_t gamma=1.0);

		virtual ~PolyKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type POLY
		 */
		virtual EKernelType get_kernel_type() { return K_POLY; }

		/** return the kernel's name
		 *
		 * @return name Poly
		 */
		virtual const char* get_name() const { return "PolyKernel"; };

		/** @return degree of kernel */
		virtual int32_t get_degree() { return degree; }

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
		/** parameter trading off the influence of higher-order versus lower-order terms in the polynomial */
		float64_t m_c;
		/** gamma scaler */
		float64_t m_gamma;
};
}
#endif /* _POLYKERNEL_H__ */
