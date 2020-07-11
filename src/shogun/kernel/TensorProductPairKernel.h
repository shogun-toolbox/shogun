/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _TPPKKERNEL_H___
#define _TPPKKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Computes the Tensor Product Pair Kernel (TPPK).
 *
 * Formally, it computes
 *
 * \f[
 * k_{\mbox{tppk}}(({\bf a},{\bf b}), ({\bf c},{\bf d}))= k({\bf a}, {\bf c})\cdot k({\bf b}, {\bf c}) + k({\bf a},{\bf d})\cdot k({\bf b}, {\bf c})
 * \f]
 *
 * It is defined on pairs of inputs and a subkernel \f$k\f$. The subkernel has
 * to be given on initialization. The pairs are specified via indizes (ab)using
 * 2-dimensional integer features.
 *
 * Its feature space \f$\Phi_{\mbox{tppk}}\f$ is the tensor product of the
 * feature spaces of the subkernel \f$k(.,.)\f$ on its input.
 *
 * It is often used in bioinformatics, e.g., to predict protein-protein interactions.
 */
class TensorProductPairKernel: public DotKernel
{
	public:
		/** default constructor  */
		TensorProductPairKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param subkernel the subkernel
		 */
		TensorProductPairKernel(int32_t size, std::shared_ptr<Kernel> subkernel);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param subkernel the subkernel
		 */
		TensorProductPairKernel(const std::shared_ptr<DenseFeatures<int32_t>>& l, const std::shared_ptr<DenseFeatures<int32_t>>& r, std::shared_ptr<Kernel> subkernel);

		~TensorProductPairKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** return what type of kernel we are
		 *
		 * @return kernel type TPPK
		 */
		EKernelType get_kernel_type() override { return K_TPPK; }

		/* register the parameters
		 */
		void register_params() override;

		/** return the kernel's name
		 *
		 * @return name TPPK
		 */
		const char* get_name() const override { return "TensorProductPairKernel"; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		EFeatureClass get_feature_class() override { return C_DENSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return int32_t feature type
		 */
		EFeatureType get_feature_type() override { return F_INT; }

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
		/** the subkernel */
		std::shared_ptr<Kernel> subkernel;
};
}
#endif /* _TPPKKERNEL_H__ */
