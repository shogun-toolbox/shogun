/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Alexander Binder
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef PYRAMIDCHI2_H_
#define PYRAMIDCHI2_H_

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

//TODO: port to CCombinedKernel (if it is the appropriate) as the pyramid is a weighted linear combination of kernels

///Pyramid Kernel over Chi2 matched histograms
class CPyramidChi2 : public CSimpleKernel<float64_t>
{
public:

	/** constructor
	 *
	 * @param size size
	 * @param width2 width2
	 * @param pyramidlevels2 pyramidlevels2
	 * @param numlevels2 numlevels2
	 * @param numbinsinhistogram2 numbinsinhistogram2
	 * @param weights2 weights2
	 * @param numweights2 numweights2
	 */
	CPyramidChi2(
		int32_t size, float64_t width2,
		int32_t* pyramidlevels2, int32_t numlevels2,
		int32_t  numbinsinhistogram2, float64_t* weights2, int32_t numweights2);

	/** constructor
	 *
	 * @param l features lhs
	 * @param r features rhs
	 * @param size size
	 * @param width2 width2
	 * @param pyramidlevels2 pyramidlevels2
	 * @param numlevels2 numlevels2
	 * @param numbinsinhistogram2 numbinsinhistogram2
	 * @param weights2 weights2
	 * @param numweights2 numweights2
	 */
	CPyramidChi2(
		CRealFeatures* l, CRealFeatures* r, int32_t size, float64_t width2,
		int32_t* pyramidlevels2, int32_t numlevels2,
		int32_t  numbinsinhistogram2, float64_t* weights2, int32_t numweights2);

	/** init
	 *
	 * @param l features lhs
	 * @param r reatures rhs
	 */
	virtual bool init(CFeatures* l, CFeatures* r);


	virtual ~CPyramidChi2();

	/** cleanup */
	virtual void cleanup();

	/* load kernel init_data
	 *
	 * @param src source file to load from
	 * @return if loading was successful
	 */
	virtual bool load_init(FILE* src);

	/** save kernel init_data
	 *
	 * @param dest destination file to save to
	 * @return if saving was succesful
	 */
	virtual bool save_init(FILE* dest);

	/** return what type of kernel we are Linear,Polynomial, Gaussian,... */
	virtual EKernelType get_kernel_type()
	{
		//preliminary output
		return K_PYRAMIDCHI2;
	}

	/** return the name of a kernel */
	virtual const char* get_name() { return "PyramidoverChi2"; }

	/** sets standard weights */
	void setstandardweights();

	/** performs a weak check, does not test for correct feature length */
	bool sanitycheck_weak();

protected:
	/** compute kernel function for features a and b
	 *
	 * @param idx_a index of feature vector a
	 * @param idx_b index of feature vector b
	 * @return computed kernel function
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

protected:
	/** width */
	float64_t width;
	/** pyramidlevels */
	int32_t* pyramidlevels;

	/** length of vector pyramidlevels */
	int32_t numlevels;
	/** numbinsinhistogram */
	int32_t numbinsinhistogram;
	/** weights */
	float64_t* weights;

	/** length of vector weights */
	int32_t numweights;
	//bool sanitycheckbit;
};

#endif /*PYRAMIDCHI2_H_*/
