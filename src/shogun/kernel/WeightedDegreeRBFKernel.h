#ifndef _WEIGHTEDDEGREERBFKERNEL_H___
#define _WEIGHTEDDEGREERBFKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief weighted degree RBF kernel */
class WeightedDegreeRBFKernel: public DotKernel
{
	public:
		/** default constructor
		 *
		 */
		WeightedDegreeRBFKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 * @param degree degree
		 * @param nof_properties number of properties per amino acid
		 */
		WeightedDegreeRBFKernel(int32_t size, float64_t width, int32_t degree, int32_t nof_properties);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param degree degree
		 * @param nof_properties number of properties per amino acid
		 * @param size cache size
		 */
		WeightedDegreeRBFKernel(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r,
			float64_t width, int32_t degree, int32_t nof_properties, int32_t size=10);

		~WeightedDegreeRBFKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** return what type of kernel we are
		 *
		 * @return kernel type UNKNOWN
		 */
		//virtual EKernelType get_kernel_type() { return K_UNKNOWN; }
		EKernelType get_kernel_type() override { return K_WEIGHTEDDEGREERBF; }

		/** return the kernel's name
		 *
		 * @return name Gaussian
		 */
		const char* get_name() const override { return "WeightedDegreeRBFKernel"; }


		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		EFeatureClass get_feature_class() override { return C_DENSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return float64_t feature type
		 */
		EFeatureType get_feature_type() override { return F_DREAL; }


		/** Set width
		 *
		 * @param w new width
		 */
		void set_width(float64_t w) { width=w; }

		/** Get width
		 *
		 * @return width
		 */
		float64_t get_width() { return width; }

		/** set degree
		 *
		 * @param deg new degree
		 * @return if setting was successful
		 */
		inline bool set_degree(int32_t deg) { degree=deg; return true; }

		/** get degree
		 *
		 * @return degree
		 */
		inline int32_t get_degree() { return degree; }

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

		/** init degree weights
		 *
		 * @return if initialization was successful
		 */
		bool init_wd_weights();

	protected:
		/** width */
		float64_t width;

		/** degree */
		int32_t degree;

		/** number of properties per amino acid */
		int32_t nof_properties;

		/** weights
		 */
		float64_t* weights;

	private:
		/** register parameters */
		void register_params() override;

};
}
#endif /* _WEIGHTEDDEGREERBFKERNEL_H__ */
