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
	class CMaternKernel: public CDistanceKernel 
	{
		public:
			/** default constructor */
			CMaternKernel();

			/** constructor
			 *
			 * @param size cache size
			 * @param width the kernel width
			 * @param order the order of the bessel function of the second kind
			 * @param dist distance to be used
			 */
			CMaternKernel(int32_t size, float64_t width, float64_t order, CDistance* dist);

			/** destructor */
			~CMaternKernel();

			/** cleanup instance */
			void cleanup();

			/** return the kernel's name
			 *
			 * @return name
			 */
			virtual const char* get_name() const
			{
				return "MaternKernel";
			}

			/** initialize kernel
			 *
			 * @param l features of left-hand side
			 * @param r features of right-hand side
			 * @return if initializing was successful
			 */
			virtual bool init(CFeatures* l, CFeatures* r);

		protected:
			float64_t compute(int32_t idx_a, int32_t idx_b);

		private:
			void init();

			float64_t nu;
	};
}