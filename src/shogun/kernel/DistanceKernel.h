/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#ifndef _DISTANCEKERNEL_H___
#define _DISTANCEKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{
	class Distance;

/** @brief The Distance kernel takes a distance as input.
 *
 * It turns a distance into something kernel like by computing
 *
 * \f[
 *     k({\bf x}, {\bf x'}) = e^{-\frac{dist({\bf x}, {\bf x'})}{width}}
 * \f]
 */
class DistanceKernel: public Kernel
{
	public:
		/** default constructor  */
		DistanceKernel();

		/** constructor
		 *
		 * @param cache cache size
		 * @param width width
		 * @param dist distance
		 */
		DistanceKernel(int32_t cache, float64_t width, std::shared_ptr<Distance> dist);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param dist distance
		 */
		DistanceKernel(
			std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t width, std::shared_ptr<Distance> dist);

		virtual ~DistanceKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** register the parameters (serialization support)
		 *
		*/
		virtual void register_params();

		/** return what type of kernel we are
		 *
		 * @return kernel type DISTANCE
		 */
		virtual EKernelType get_kernel_type() { return K_DISTANCE; }
		/** return feature type the kernel can deal with
		 *
		 * @return feature type of distance used
		 */
		virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class of distance used
		 */
		virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }

		/** return the kernel's name
		 *
		 * @return name Distance
		 */
		virtual const char* get_name() const { return "DistanceKernel"; }

		/** set the kernel's width
		 *
		 * @param w kernel width
		 */
		virtual void set_width(float64_t w)
		{
			width=w;
		}

		/** return the kernel's width
		 *
		 * @return kernel width
		 */
		virtual float64_t get_width() const
		{
			return width;
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
		float64_t compute(int32_t idx_a, int32_t idx_b);

		/** distance */
		std::shared_ptr<Distance> distance;
		/** width */
		float64_t width;
};
}
#endif /* _DISTANCEKERNEL_H__ */
