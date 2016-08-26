/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written by Haipeng Wang 2011
 *
 * This is a simple class of Parsen Window estimator based on Gaussian kernel function.
 * The core estimating function is f(x)=(1/n)*(1/pow(sqrt(2*Pi)*h,d))*sum_{i=1}^{n}{(exp(-0.5/(h*h) * ||x-x_{i}||_{2}^{2}))}
 */
 
#ifndef __ParsenWindow_H__
#define __ParsenWindow_H__

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
class ParsenWindow
{
public:
	/** Construct ParsenWindow Class */ 
	ParsenWindow();

	/** Destructor */
	virtual ~ParsenWindow();

	/** ParsenWindow Initialization
	 *	This inilization should be run before calculating density.
	 *	nFeaDim: Feature dimension.
	 *	nTrainPoints: Number of Training Samples.
	 *	fWindowWidth: Window Width.
	 *  ppTrainingPoints: The ith pointer ppTrainingPoints[i] points to the address where the ith training sample is stored.
	 */
	bool init(int32_t nFeaDim, int32_t nTrainPoints, float64_t fWindowWidth, float64_t** ppTrainingPoints);

	/** Calculate Gassuian kernel distance between two vectors.
	 *	the core function is dis(x,y)=(1/pow(sqrt(2*Pi)*h,d))*(exp(-0.5/(h*h) * ||x-y||_{2}^{2})
	 *  This function returns the Gassuian kernel distance;
	 */
	float64_t GaussianProb(const float64_t* Vector1, const float64_t* Vector2);
	
	/** Calculate all the densities of all the testing samples based on the training samples
	 *  pDensity: address where the calculated densities are stored
	 *  ppTestingPoints: The ith pointer ppTestingPoints[i] points to the address where the ith testing sample is stored.
	 *  nTestingPoints: number of testing samples
	 *  nFeaDim_Test: testing sample feature dimension
	 */
	void CalculateDensity(float64_t* const pDensity, const float64_t** ppTestingPoints, int32_t nTestingPoints, int32_t nFeaDim_Test);
	
protected:
	/** Training sample feature dimension */
	int32_t	nVecSize;

	/** number of training samples */
	int32_t nTrainingSample;

	/** window width
	 *  Since we use Gaussian kernel function as the window function, the window width is used as the kernel width.
	 */
	float64_t fWidth;

	/** Pointers to the pointers of training Data
	 *  The ith pointer ppTrainingData[i] points to the address where the ith training sample is stored.
	 *  We don't copy the training data into this class, but just pass this pointer in.
	 */
	float64_t** ppTrainingData;

	/** Guassian kernel const = 1/(pow(sqrt(2*pi)*h),d) */
	float64_t fGaussianConst;
};
}

#endif