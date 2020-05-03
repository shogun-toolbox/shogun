/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/kernel/DistanceKernel.h>
#include <shogun/kernel/Kernel.h>


namespace shogun {


	/**
	 * @brief the class Matérn kernel 
	 *
	 * It is defined as
	 * \f[
	 * k(x, y) = \frac{2 ^ {\nu - 1}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{D(x, y)}{\rho}\Bigg)^\nu + N_{\nu}\Bigg(\sqrt{2\nu}\frac{D(x, y)}{\rho}\Bigg)
	 * \f]
	 * Where:
	 * 		\f$N_{\nu}\f$ is the Bessel function of the second kind with order \f$\nu\f$,
	 *		\f$\rho\f$ is the kernel width, and
	 *		\f$\Gamma\f$ is the gamma function
	 **/
	class MaternKernel: public DistanceKernel 
	{
		public:
			/** default constructor */
			MaternKernel();

			/** constructor
			 *
			 * @param size cache size
			 * @param width the kernel width
			 * @param nu the order of the bessel function of the second kind
			 * @param dist distance to be used
			 */
			MaternKernel(int32_t size, float64_t width, float64_t nu, const std::shared_ptr<Distance>& dist);

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

			/** initialize kernel
			 *
			 * @param l features of left-hand side
			 * @param r features of right-hand side
			 * @return if initializing was successful
			 */
			bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

			/** return derivative with respect to specified parameter
			 *
			 * @param param the parameter
			 * @param index the index of the element if parameter is a vector
			 *
			 * @return gradient with respect to parameter
			 */
			SGMatrix<float64_t> get_parameter_gradient(Parameters::const_reference param, index_t index=-1) override;

			protected:
				float64_t compute(int32_t idx_a, int32_t idx_b) override;

			private:
				void init();

				float64_t nu = 1.5;
	};
}