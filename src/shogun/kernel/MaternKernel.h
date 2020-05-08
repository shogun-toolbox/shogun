/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef MATERNKERNEL_H
#define MATERNKERNEL_H

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/ShiftInvariantKernel.h>

namespace shogun
{

	/**
	 * @brief The Matérn kernel
	 *
	 * It is defined as
	 * \f[
	 * k(x, y) = \frac{2 ^ {\nu - 1}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{D(x,
	 *y)}{\rho}\Bigg)^\nu + N_{\nu}\Bigg(\sqrt{2\nu}\frac{D(x, y)}{\rho}\Bigg)
	 * \f]
	 * Where:
	 * 		\f$N_{\nu}\f$ is the order of the modified Bessel function of the second kind
	 *		\f$\rho\f$ is the kernel width
	 *		\f$\Gamma\f$ is the gamma function
	 **/
	class MaternKernel : public ShiftInvariantKernel
	{
	public:
		/** default constructor */
		MaternKernel();

		/** constructor
		 *
		 * @param width the kernel width
		 * @param nu the order of the bessel function of the second kind
		 * @param dist distance to be used
		 */
		MaternKernel(float64_t width, float64_t nu);

		/** constructor
		 *
		 * @param size cache size
		 * @param width the kernel width
		 * @param nu the order of the bessel function of the second kind
		 * @param dist distance to be used
		 */
		MaternKernel(int32_t size, float64_t width, float64_t nu);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param width the kernel width
		 * @param nu the order of the bessel function of the second kind
		 * @param dist distance to be used
		 */
		MaternKernel(
		    const std::shared_ptr<Features>& l,
		    const std::shared_ptr<Features>& r, int32_t size, float64_t width,
		    float64_t nu);

		/** destructor */
		~MaternKernel();

		/** return the kernel's name
		 *
		 * @return name
		 */
		const char* get_name() const override
		{
			return "MaternKernel";
		}

		/** clean up kernel */
		void cleanup() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool
		init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** return derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		SGMatrix<float64_t> get_parameter_gradient(
		    Parameters::const_reference param, index_t index = -1) override;

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIAN
		 */
		EKernelType get_kernel_type() override
		{
			return K_MATERN;
		}

		/** @return feature type of distance used */
		EFeatureType get_feature_type() override
		{
			return F_ANY;
		}

		/** @return feature class of distance used */
		EFeatureClass get_feature_class() override
		{
			return C_ANY;
		}

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

	private:
		/* order of the Bessel function of the second kind */
		float64_t m_nu = 1.5;
		/* the kernel width */
		float64_t m_width = 1.0;
	};
}

#endif