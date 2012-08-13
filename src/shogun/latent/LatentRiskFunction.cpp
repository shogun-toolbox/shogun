/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/latent/LatentRiskFunction.h>

using namespace shogun;

CLatentRiskFunction::CLatentRiskFunction()
	: CRiskFunction()
{
}

CLatentRiskFunction::~CLatentRiskFunction()
{

}

void CLatentRiskFunction::risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	ASSERT(data != NULL);
	ASSERT(R != NULL);
	ASSERT(subgrad != NULL);
	ASSERT(W != NULL);

	*R = 0;
	
}
