/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Haipeng Wang
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "ParzenWindow.h"

using namespace shogun;

ParzenWindow::ParzenWindow()
	: CDistribution()
{
	m_vec_size = 0;
	m_TrainingSample_num = 0;
	m_fWidth = 0.0;
	m_ppTrainingData = NULL;
	m_fGaussianConst = 0.0;
}

/** Since we don't store any samples in this class, so the Destructor just reset the pointer */
ParzenWindow::~ParzenWindow()
{
	if(m_vec_size * m_TrainingSample_num > 0)
	{
		m_ppTrainingData = NULL;
	}
}

/** Initilization: 
 * 1) set the basic config and pass in the pointer to the training data
 * 2) calculate the Gaussian constant.
 */ 
bool ParzenWindow::init(int32_t nFeaDim, int32_t nTrainPoints, float64_t fWindowWidth, float64_t** ppTrainingPoints)
{
	if(nFeaDim <= 0 || nTrainPoints <= 0 || fWindowWidth  <= 0 || ppTrainingPoints == NULL)
	{
		return false;
	}
	else
	{
		m_vec_size = nFeaDim;
		m_TrainingSample_num = nTrainPoints;
		m_fWidth = fWindowWidth;
		m_ppTrainingData = ppTrainingPoints;
		m_fGaussianConst = pow(sqrt(DPI)*m_fWidth, m_vec_size);
		m_fGaussianConst = 1/m_fGaussianConst;
		return true;
	}
}

/** Calculate Gassuian kernel distance between two vectors. */
float64_t ParzenWindow::GaussianProb(const float64_t* Vector1, const float64_t* Vector2)
{
	int32_t i = 0;
	float64_t dis = 0;
	for(i = 0; i <m_vec_size; i++)
	{
		dis += (Vector1[i]-Vector2[i]) * (Vector1[i]-Vector2[i]);
	}
	dis = -1*dis/(2*m_fWidth*m_fWidth);
	if(dis < MinExpArg)
	{
		dis = 0;
	}
	else
	{
		dis = exp(dis);
	}
	dis = dis*m_fGaussianConst;
	return dis;
}

/** Calculate all the densities of all the testing samples based on the training samples */
void ParzenWindow::CalculateDensity(float64_t* const pDensity, const float64_t** ppTestingPoints, int32_t nTestingPoints, int32_t nFeaDim_Test)
{
	if(nFeaDim_Test != m_vec_size)
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
			pDensity[i] += GaussianProb(ppTestingPoints[i], m_ppTrainingData[j]);
		}
		pDensity[i] = pDensity[i]/m_TrainingSample_num;
	}
}
