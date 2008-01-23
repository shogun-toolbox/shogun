/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"


#ifndef _DISTANCEKERNEL_H___
#define _DISTANCEKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "distance/Distance.h"

class CDistanceKernel: public CKernel
{
public:
	/* Constructors */
	CDistanceKernel(INT cache, DREAL width, CDistance* dist);
	CDistanceKernel(CFeatures *l, CFeatures *r, DREAL width, CDistance* dist);
	virtual ~CDistanceKernel();

	/* Init and cleanup functions */
	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/* Identification functions */
	inline virtual EKernelType get_kernel_type() { return K_DISTANCE; }
	inline virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }
	inline virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }
	inline virtual const CHAR* get_name() { return distance->get_name(); }

	/* Load and save functions */
	bool load_init(FILE* src);
	bool save_init(FILE* dest);

protected:
	/* Kernel function */
	DREAL compute(INT idx_a, INT idx_b);

private:
	CDistance* distance;
	DREAL width;
};

#endif /* _DISTANCEKERNEL_H__ */

