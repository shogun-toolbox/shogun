/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANKERNEL_H___
#define _GAUSSIANKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/SimpleFeatures.h"

namespace shogun
{
/** @brief The well known Gaussian kernel (swiss army knife for SVMs)
 * on dense real valued features.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class CGaussianKernel: public CSimpleKernel<float64_t>
{
	public:
		/** default constructor
		 *
		 */
		CGaussianKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CGaussianKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		CGaussianKernel(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r,
			float64_t width, int32_t size=10);

		virtual ~CGaussianKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

		/** return the kernel's name
		 *
		 * @return name Gaussian
		 */
		inline virtual const char* get_name() const { return "GaussianKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

#ifdef HAVE_BOOST_SERIALIZATION
    private:

        friend class ::boost::serialization::access;
        template<class Archive>
            void serialize(Archive & ar, const unsigned int archive_version)
            {

                SG_DEBUG("archiving GaussianKernel\n");

                ar & ::boost::serialization::base_object<CSimpleKernel<float64_t> >(*this);

                ar & width;

                SG_DEBUG("done with GaussianKernel\n");

            }
#endif //HAVE_BOOST_SERIALIZATION


	protected:
		/** width */
		float64_t width;
};
}
#endif /* _GAUSSIANKERNEL_H__ */
