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
		/** default constructor  */
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

		virtual ~DiagKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

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

		/** return what type of kernel we are
		 *
		 * @return kernel type CUSTOM
		 */
		virtual EKernelType get_kernel_type() { return K_DIAG; }

		/** return the kernel's name
		 *
		 * @return name Custom
		 */
		virtual const char* get_name() const { return "DiagKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			if (idx_a==idx_b)
				return diag;
			else
				return 0;
		}
	private:
		void init();

	protected:
		/** diagonal */
		float64_t diag;
};
}
#endif /* _DIAGKERNEL_H__ */
