/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preproc/MDS.h"
#include "lib/lapack.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CMDS::CMDS() : CSimplePreProc<float64_t>()
{
}

CMDS::~CMDS()
{
}

bool CMDS::init(CFeatures* data)
{
	return true;
}

void CMDS::cleanup()
{

}

float64_t* CMDS::apply_to_feature_matrix(CFeatures* f)
{
	return 0;
}

float64_t* CMDS::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	return 0;
}
