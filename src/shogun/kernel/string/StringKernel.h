/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _STRINGKERNEL_H___
#define _STRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
/** @brief Template class StringKernel, is the base class of all String Kernels.
 *
 * For a (very complex) example see e.g. CWeightedDegreeStringKernel
 *
 */
template <class ST> class StringKernel : public Kernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		StringKernel(int32_t cachesize=0) : Kernel(cachesize) {}

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		StringKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r) : Kernel(10)
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

			ASSERT(l->get_feature_class()==C_STRING)
			ASSERT(r->get_feature_class()==C_STRING)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())

			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class STRING
		 */
		virtual EFeatureClass get_feature_class() { return C_STRING; }

		/** return feature type the kernel can deal with
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "StringKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0;
};

template<> inline EFeatureType StringKernel<float64_t>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType StringKernel<uint64_t>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType StringKernel<int32_t>::get_feature_type() { return F_INT; }

template<> inline EFeatureType StringKernel<uint16_t>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType StringKernel<int16_t>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType StringKernel<uint8_t>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType StringKernel<char>::get_feature_type() { return F_CHAR; }
}
#endif /* _STRINGKERNEL_H__ */

