/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Sergey Lisitsyn
 */


#ifndef _CIRCULARKERNEL_H__
#define _CIRCULARKERNEL_H__

#include <shogun/kernel/ShiftInvariantKernel.h>

namespace shogun
{

/** @brief Circular kernel
 *
 * Formally described as
 *
 * \f[
 *     k(x, y) = \frac{2}{\pi}*\arccos(\frac{-||{\bf x}-{\bf x'}||}{\sigma}) - \frac{2}{\pi}*(\frac{||{\bf x}-{\bf x'}||}{\sigma})*\sqrt{1-(\frac{||{\bf x}-{\bf x'}||}{\sigma})^2}
 *
 *     \mbox{if}~ ||x-y|| \leq \sigma \mbox{, zero otherwise}
 * \f]
 *
 */

class CircularKernel: public ShiftInvariantKernel
{
	public:
	/** default constructor */
	CircularKernel();

	/** constructor
	 *
	 * @param size cache size
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	CircularKernel(int32_t size, float64_t sigma, std::shared_ptr<Distance> dist);

	/** constructor
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	CircularKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sigma, std::shared_ptr<Distance> dist);

	/** initialize kernel with features
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @return true if successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/**
	 * @return kernel type
	 */
	EKernelType get_kernel_type() override { return K_CIRCULAR; }

	/**
	 * @return type of features
	 */
	EFeatureType get_feature_type() override { return distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	EFeatureClass get_feature_class() override { return distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	const char* get_name() const override { return "CircularKernel"; }

	/** set the kernel's sigma
	 *
	 * @param s kernel sigma
	 */
	virtual void set_sigma(float64_t s)
	{
		sigma=s;
	}

	/** return the kernel's sigma
	 *
	 * @return kernel sigma
	 */
	virtual float64_t get_sigma() const
	{
		return sigma;
	}

	~CircularKernel() override;

	/** Can (optionally) be overridden to post-initialize some
	 *  member variables which are not PARAMETER::ADD'ed.  Make
	 *  sure that at first the overridden method
	 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	void load_serializable_post() override;

	/**
	 * compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 */
	float64_t compute(int32_t idx_a, int32_t idx_b) override;

private:
	void init();

protected:

	/** distance */
	std::shared_ptr<Distance> distance;

	/** width */
	float64_t sigma;

};
}

#endif /* _CIRCULARKERNEL_H__ */
