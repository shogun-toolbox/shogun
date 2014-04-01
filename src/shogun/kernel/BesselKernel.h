/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Ziyuan Lin
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef BESSELKERNEL_H_
#define BESSELKERNEL_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/DistanceKernel.h>

namespace shogun
{
class CDistance;
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
class CBesselKernel: public CDistanceKernel
{
	public:
		/** default constructor */
		CBesselKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param order the order of the bessel function
		 * @param width the kernel width
		 * @param degree the kernel degree
		 * @param dist distance to be used
		 */
		CBesselKernel(int32_t size, float64_t order,
				float64_t width, int32_t degree,
				CDistance* dist);

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
		CBesselKernel(CFeatures* l, CFeatures* r,
				float64_t order, float64_t width, int32_t degree,
				CDistance* dist, int32_t size=10);

		/**
		 * clean up kernel
		 */
		virtual ~CBesselKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** cleanup */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type Bessel
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_BESSEL;
		}

		/**
		 * @return type of features
		 */
		virtual EFeatureType get_feature_type()
		{
			return distance->get_feature_type();
		}

		/**
		 * @return class of features
		 */
		virtual EFeatureClass get_feature_class()
		{
			return distance->get_feature_class();
		}

		/** return the kernel's name
		 *
		 * @return name Bessel
		 */
		virtual const char* get_name() const
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
		virtual void set_width(float64_t tau)
		{
			width = tau;
		}

		/** return the kernel's width
		 *
		 * @return kernel width
		 */
		virtual float64_t get_width() const
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
		float64_t compute(int32_t idx_a, int32_t idx_b);

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
