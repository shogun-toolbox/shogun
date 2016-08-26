/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written by Haipeng Wang 2011
 *
 * This is  a simple class of Parsen Window estimator based on Gaussian kernel function.
 * The core estimating function is f(x)=(1/n)*(1/pow(sqrt(2*Pi)*h,d))*sum_{i=1}^{n}{(exp(-0.5/(h*h) * ||x-x_{i}||_{2}^{2}))}
 */
 
#include "ParsenWindow.h"

using namespace shogun;

ParsenWindow::ParsenWindow()
{
	nVecSize = 0;
	nTrainingSample = 0;
	fWidth = 0.0;
	ppTrainingData = NULL;
	fGaussianConst = 0.0;
}

/** Since we don't store any samples in this class, so the Destructor just reset the pointer */
ParsenWindow::~ParsenWindow()
{
	if(nVecSize * nTrainingSample > 0)
	{
		ppTrainingData = NULL;
	}
}

/** Initilization: 
 * 1) set the basic config and pass in the pointer to the training data
 * 2) calculate the Gaussian constant.
 */ 
bool ParsenWindow::init(int32_t nFeaDim, int32_t nTrainPoints, float64_t fWindowWidth, float64_t** ppTrainingPoints)
{
	if(nFeaDim <= 0 || nTrainPoints <= 0 || fWindowWidth  <= 0 || ppTrainingPoints == NULL)
	{
		return false;
	}else
	{
		nVecSize = nFeaDim;
		nTrainingSample = nTrainPoints;
		fWidth = fWindowWidth;
		ppTrainingData = ppTrainingPoints;
		fGaussianConst = pow(sqrt(DPI)*fWidth, nVecSize);
		fGaussianConst = 1/fGaussianConst;
		return true;
	}
}

/** Calculate Gassuian kernel distance between two vectors. */
float64_t ParsenWindow::GaussianProb(const float64_t* Vector1, const float64_t* Vector2)
{
	int32_t i = 0;
	float64_t dis = 0;
	for(i = 0; i <nVecSize; i++)
	{
		dis += (Vector1[i]-Vector2[i]) * (Vector1[i]-Vector2[i]);
	}
	dis = -1*dis/(2*fWidth*fWidth);
	if(dis < MinExpArg)
	{
		dis = 0;
	}else
	{
		dis = exp(dis);
	}
	dis = dis/fGaussianConst;
	return dis;
}

/** Calculate all the densities of all the testing samples based on the training samples */
void ParsenWindow::CalculateDensity(float64_t* const pDensity, const float64_t** ppTestingPoints, int32_t nTestingPoints, int32_t nFeaDim_Test)
{
	if(nFeaDim_Test != nVecSize)
	{
		SG_ERROR("Feature Dimensions for Training samples and Testing samples should be the same\n");
	}
	if(pDensity == NULL)
	{
		SG_ERROR("Density vector space should be allocated\n");
	}
	if(ppTestingPoints == NULL)
	{
		SG_ERROR("Testing feature vectors should be provided\n");
	}
	if(nTestingPoints == 0)
	{
		SG_ERROR("No Testing Samples\n");
	}
	if(ppTrainingData == NULL)
	{
		SG_ERROR("Training feature vectors should be provided\n");
	}
	int32_t i = 0;
	int32_t j = 0;
	for(i = 0; i <nTestingPoints; i++)
	{
		pDensity[i] = 0.0;
		for(j = 0; j < nTrainingSample; j++)
		{
			pDensity[i] += GaussianProb(ppTestingPoints[i], ppTrainingData[j]);
		}
		pDensity[i] = pDensity[i]/nTrainingSample;
	}
}
