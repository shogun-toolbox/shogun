/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Wu Lin
 */

#ifndef _DOTKERNEL_H___
#define _DOTKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief Template class DotKernel is the base class for kernels working on
 * DotFeatures.
 *
 * DotFeatures are features supporting operations like dot product, dot product
 * with a dense vector and addition to a dense vector. Therefore several dot
 * product based kernels derive from this class (cf., e.g., LinearKernel)
 *
 * \sa DotFeatures
 */
class DotKernel : public Kernel
{
	public:
		/** default constructor
		 *
		 */
		DotKernel() : Kernel() {}

		/** constructor
		 *
		 * @param cachesize cache size
		 */
		DotKernel(int32_t cachesize) : Kernel(cachesize) {}

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		DotKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r) : Kernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization
		 *  constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *
		 *  @param l features for left-hand side
		 *  @param r features for right-hand side
		 *  @return if init was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
		{
			Kernel::init(l,r);
			init_auto_params();

			ASSERT(l->has_property(FP_DOT))
			ASSERT(r->has_property(FP_DOT))
			ASSERT(l->get_feature_type() == r->get_feature_type());
			if (l->support_compatible_class())
			{
				require(l->get_feature_class_compatibility(r->get_feature_class()),
					"Right hand side of features ({}) must be compatible with left hand side features ({})",
					l->get_name(), r->get_name());
			}
			else
			{
				require(l->get_feature_class()==r->get_feature_class(),
					"Right hand side of features ({}) must be compatible with left hand side features ({})",
					l->get_name(), r->get_name());
			}

			if ( (std::static_pointer_cast<DotFeatures>(l))->get_dim_feature_space() != (std::static_pointer_cast<DotFeatures>(r))->get_dim_feature_space() )
			{
				error("train or test features #dimension mismatch (l:{} vs. r:{})",
						(std::static_pointer_cast<DotFeatures>(l))->get_dim_feature_space(),
						(std::static_pointer_cast<DotFeatures>(r))->get_dim_feature_space());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * dot kernel returns unknown since features can be based on anything
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return feature type the kernel can deal with
		 *
		 * dot kernel returns unknown since features can be based on anything
		 *
		 * @return ANY feature type
		 */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "DotKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0 ;

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
			return (std::static_pointer_cast<DotFeatures>(lhs))->dot(idx_a, (std::static_pointer_cast<DotFeatures>(rhs)), idx_b);
		}
};
}
#endif /* _DOTKERNEL_H__ */

