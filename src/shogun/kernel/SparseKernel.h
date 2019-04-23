/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SPARSEKERNEL_H___
#define _SPARSEKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
/** @brief Template class SparseKernel, is the base class of kernels working on sparse
 * features.
 *
 * See e.g. the CSparseGaussianKernel for an example.
 */
template <class ST> class SparseKernel : public Kernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		SparseKernel(int32_t cachesize) : Kernel(cachesize) {}

		/** constructor
		 *
		 * @param l features for left-hand side
		 * @param r features for right-hand side
		 */
		SparseKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r) : Kernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
		{
			Kernel::init(l,r);

			ASSERT(l->get_feature_class()==C_SPARSE)
			ASSERT(r->get_feature_class()==C_SPARSE)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())

			auto sf_lhs = lhs->as<SparseFeatures<ST>>();
			auto sf_rhs = rhs->as<SparseFeatures<ST>>();

			if (sf_lhs->get_num_features() != sf_rhs->get_num_features())
			{
				SG_ERROR("train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						sf_lhs->get_num_features(),sf_rhs->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SPARSE
		 */
		virtual EFeatureClass get_feature_class() { return C_SPARSE; }

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
			return "SparseKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0;
};

template<> inline EFeatureType SparseKernel<float64_t>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType SparseKernel<uint64_t>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType SparseKernel<int32_t>::get_feature_type() { return F_INT; }

template<> inline EFeatureType SparseKernel<uint16_t>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType SparseKernel<int16_t>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType SparseKernel<uint8_t>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType SparseKernel<char>::get_feature_type() { return F_CHAR; }
}
#endif /* _SPARSEKERNEL_H__ */
