/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Haipeng Wang
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __ParzenWindow_H__
#define __ParzenWindow_H__

#ifndef PI
/** just in case PI is not defined */
#define PI   3.14159265358979
#endif

/** 2*PI */
#define DPI  6.28318530717959

/** lowest exp() arg. If arg <  MinExpArg, exp(arg) = 0*/
#define MinExpArg (-700.0)

#include <stdio.h>
#include <math.h>
#include "lib/Mathematics.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/config.h"
#include "features/Features.h"
#include "distributions/Distribution.h"


namespace shogun
{
class ParzenWindow::public CDistribution
{
public:
	/** default constructor */ 
	ParzenWindow();

	/** destructor */
	virtual ~ParzenWindow();

	/** ParzenWindow Initialization
	 *	This inilization should be run before calculating density.
	 *	@param nFeaDim Feature dimension.
	 *	@param nTrainPoints Number of Training Samples.
	 *	@param fWindowWidth Window Width.
	 *  @param ppTrainingPoints The ith pointer ppTrainingPoints[i] points to the address where the ith training sample is stored.
	 *  @return true if successful
	 */
	bool init(int32_t nFeaDim, int32_t nTrainPoints, float64_t fWindowWidth, float64_t** ppTrainingPoints);

	/** Calculate Gassuian kernel distance between two vectors.
	 *	the core function is dis(x,y)=(1/pow(sqrt(2*Pi)*h,d))*(exp(-0.5/(h*h) * ||x-y||_{2}^{2})
	 *  @return true if successful the Gassuian kernel distance;
	 */
	float64_t GaussianProb(const float64_t* Vector1, const float64_t* Vector2);
	
	/** Calculate all the densities of all the testing samples based on the training samples
	 *  @param pDensity address where the calculated densities are stored
	 *  @param ppTestingPoints The ith pointer ppTestingPoints[i] points to the address where the ith testing sample is stored.
	 *  @param nTestingPoints number of testing samples
	 *  @param nFeaDim_Test testing sample feature dimension
	 */
	void CalculateDensity(float64_t* const pDensity, const float64_t** ppTestingPoints, int32_t nTestingPoints, int32_t nFeaDim_Test);
	
protected:
	/** Training sample feature dimension */
	int32_t	m_vec_size;

	/** number of training samples */
	int32_t m_TrainingSample_num;

	/** window width
	 *  Since we use Gaussian kernel function as the window function, the window width is used as the kernel width.
	 */
	float64_t m_fWidth;

	/** Pointers to the pointers of training Data
	 *  The ith pointer ppTrainingData[i] points to the address where the ith training sample is stored.
	 *  We don't copy the training data into this class, but just pass this pointer in.
	 */
	float64_t** m_ppTrainingData;

	/** Guassian kernel const = 1/(pow(sqrt(2*pi)*h),d) */
	float64_t m_fGaussianConst;
};
}

#endif