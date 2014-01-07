/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * DS-Kernel implementation Written (W) 2008 Sébastien Boisvert under GPLv3
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef DISTANTSEGMENTSKERNEL_H_
#define DISTANTSEGMENTSKERNEL_H_

#include <kernel/string/StringKernel.h>

namespace shogun
{

/**
 * @brief The distant segments kernel is a string kernel,
 * which counts the number of substrings, so-called segments,
 * at a certain distance from each other.
 *
 * The implementation is taken from
 * http://www.retrovirology.com/content/5/1/110/ and
 * only adjusted to work with shogun. See that page for any details.
 *
 * Reference: Sebastien Boisvert, Mario Marchand, Francois Laviolette,
 * and Jacques Corbeil. Hiv-1 coreceptor usage prediction without
 * multiple alignments: an application of string kernels.
 * Retrovirology, 5(1):110, Dec 2008.
 */
class CDistantSegmentsKernel: public CStringKernel<char>
{
public:
	/** default constructor */
	CDistantSegmentsKernel();

	/** constructor
	 *
	 * @param size cache size
	 * @param delta \f[\delta\f]-parameter of the DS-kernel
	 * @param theta \f[\theta\f]-parameter of the DS-kernel
	 */
	CDistantSegmentsKernel(int32_t size, int32_t delta, int32_t theta);

	/** constructor
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @param size cache size
	 * @param delta \f[\delta\f]-parameter of the DS-kernel
	 * @param theta \f[\theta\f]-parameter of the DS-kernel
	 */
	CDistantSegmentsKernel(CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t size, int32_t delta, int32_t theta);

	/** initialize kernel with features
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type()
	{
		return K_DISTANTSEGMENTS;
	}

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const
	{
		return "DistantSegmentsKernel";
	}

protected:
	/**
	 * compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

private:
	/** initializes kernel parameters and registers them */
	void init();

	/**
	 * helper function taken from
	 * http://www.retrovirology.com/content/5/1/110/
	 * */
	int32_t bin(int32_t j, int32_t i);

	/**
	 * Computes the DS-kernel for the given strings and parameters.
	 * Taken from http://www.retrovirology.com/content/5/1/110/
	 * with little adjustments.
	 *
	 * @param s first string for kernel computation
	 * @param sLength length of that string
	 * @param b second string for kernel computation
	 * @param bLength length of that string
	 * @param delta_m delta parameter
	 * @param theta_m theta parameter
	 * @return computed kernel function of the given strings and parameters
	 */
	int32_t compute(char* s, int32_t sLength, char* b, int32_t bLength,
			int32_t delta_m, int32_t theta_m);

protected:
	/** delta parameter of DS-kernel */
	int32_t m_delta;

	/** theta parameter of DS-kernel */
	int32_t m_theta;


};

}

#endif /* DISTANTSEGMENTSKERNEL_H_ */
