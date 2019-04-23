/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _CONSTKERNEL_H___
#define _CONSTKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief The Constant Kernel returns a constant for all elements.
 *
 * A ``kernel'' that simply returns a single constant, i.e.
 * \f$k({\bf x}, {\bf x'})= c\f$
 *
 */
class ConstKernel: public Kernel
{
	public:
		/** default constructor  */
		ConstKernel();

		/** constructor
		 *
		 * @param c constant c
		 */
		ConstKernel(float64_t c);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param c constant c
		 */
		ConstKernel(std::shared_ptr<Features> l, std::shared_ptr<Features >r, float64_t c);

		virtual ~ConstKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type CONST
		 */
		virtual EKernelType get_kernel_type() { return K_CONST; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		virtual EFeatureType get_feature_type()
		{
			return F_ANY;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class()
		{
			return C_ANY;
		}

		/** return the kernel's name
		 *
		 * @return name Const
		 */
		virtual const char* get_name() const { return "ConstKernel"; }

	protected:
		/** compute kernel function for features a and b
		 *
		 * @param row dummy row
		 * @param col dummy col
		 * @return computed kernel function (const value)
		 */
		virtual float64_t compute(int32_t row, int32_t col)
		{
			return const_value;
		}

	private:
		void init();

	protected:
		/** const value */
		float64_t const_value;
};
}
#endif /* _CONSTKERNEL_H__ */
