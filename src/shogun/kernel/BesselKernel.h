/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser
 */

#ifndef BESSELKERNEL_H_
#define BESSELKERNEL_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/DistanceKernel.h>

namespace shogun
{
class Distance;
/** @brief the class Bessel kernel
 *
 * It is defined as
 * \f[
 * k(x, y) = \frac{J_{v}( \| x-y \|/\tau)}{ \| x-y \| ^ {-nv} }
 * \f]
 * Where:
 *		\f$J_{v}\f$ is the Bessel funcion with order \f$v\f$,
 *		\f$\tau\f$ is the kernel width, and
 *		\f$n\f$ is the kernel degree.
 * */
class BesselKernel: public DistanceKernel
{
	public:
		/** default constructor */
		BesselKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param order the order of the bessel function
		 * @param width the kernel width
		 * @param degree the kernel degree
		 * @param dist distance to be used
		 */
		BesselKernel(int32_t size, float64_t order,
				float64_t width, int32_t degree,
				std::shared_ptr<Distance> dist);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param order of the bessel function
		 * @param width the kernel width
		 * @param degree the degree
		 * @param dist distance to be used
		 * @param size cache size
		 */
		BesselKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r,
				float64_t order, float64_t width, int32_t degree,
				std::shared_ptr<Distance> dist, int32_t size=10);

		/**
		 * clean up kernel
		 */
		~BesselKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup */
		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type Bessel
		 */
		EKernelType get_kernel_type() override
		{
			return K_BESSEL;
		}

		/**
		 * @return type of features
		 */
		EFeatureType get_feature_type() override
		{
			return distance->get_feature_type();
		}

		/**
		 * @return class of features
		 */
		EFeatureClass get_feature_class() override
		{
			return distance->get_feature_class();
		}

		/** return the kernel's name
		 *
		 * @return name Bessel
		 */
		const char* get_name() const override
		{
			return "BesselKernel";
		}

		/** set the kernel's order
		 *
		 * @param v kernel order
		 */
		virtual void set_order(float64_t v)
		{
			order = v;
		}

		/** return the kernel's order
		 *
		 * @return kernel order
		 */
		virtual float64_t get_order() const
		{
			return order;
		}

		/** set the kernel's width
		 *
		 * @param tau kernel width
		 */
		void set_width(float64_t tau) override
		{
			width = tau;
		}

		/** return the kernel's width
		 *
		 * @return kernel width
		 */
		float64_t get_width() const override
		{
			return width;
		}

		/** set the kernel's degree
		 *
		 * @param n kernel degree
		 */
		virtual void set_degree(int32_t n)
		{
			degree = n;
		}

		/** return the kernel's degree
		 *
		 * @return kernel degree
		 */
		virtual int32_t get_degree() const
		{
			return degree;
		}

	protected:
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	private:
		void init();

	protected:
		/** order of the Bessel function */
		float64_t order;
		/** kernel degree */
		int32_t degree;
};

}

#endif /* BESSELKERNEL_H_ */
