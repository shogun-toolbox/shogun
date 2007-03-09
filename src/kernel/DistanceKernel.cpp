/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/DistanceKernel.h"

CDistanceKernel::CDistanceKernel(LONG cache,DREAL w ,CDistance* dist)
  :CKernel(cache),distance(dist),width(w)
{
}

CDistanceKernel::~CDistanceKernel()
{
    cleanup();
    distance=NULL;
}

bool CDistanceKernel::init(CFeatures* l, CFeatures* r)
{
    return CKernel::init(l,r) && distance->init(l,r);
}

void CDistanceKernel::cleanup()
{
}

DREAL CDistanceKernel::compute(INT idx_a, INT idx_b)
{
  return exp(-distance->distance(idx_a,idx_b)/width);
}

bool CDistanceKernel::load_init(FILE* src)
{
    return false;
}

bool CDistanceKernel::save_init(FILE* dest)
{
    return false;
}
