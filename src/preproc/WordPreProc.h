/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CWORDPREPROC__H__
#define _CWORDPREPROC__H__

#include "features/Features.h"
#include "preproc/SimplePreProc.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CWordPreProc : public CSimplePreProc<WORD>
{
public:
	CWordPreProc(const CHAR *name, const CHAR* id);
	virtual ~CWordPreProc();

	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EFeatureType get_feature_type() { return F_WORD; }
};
#endif
