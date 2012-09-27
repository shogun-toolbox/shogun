/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/features/streaming/StreamingDataGenerator.h>

using namespace shogun;

CStreamingDataGenerator::CStreamingDataGenerator() :
		CStreamingFeatures()
{
	init();
}

CStreamingDataGenerator::~CStreamingDataGenerator()
{
	delete m_model_parameters;
}

void CStreamingDataGenerator::set_model(EDataGenerator model,
		Parameter* model_parameters)
{
	m_model=model;

	/* delete old parameters */
	delete m_model_parameters;
	m_model_parameters=model_parameters;
}

void CStreamingDataGenerator::init()
{
	m_model=DG_NONE;
	m_model_parameters=NULL;

	SG_WARNING("CStreamingDataGenerator::init(): register parameters!\n");
}

