/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CDREALPREPROC__H__
#define _CDREALPREPROC__H__

#include "lib/common.h"
#include "preproc/SimplePreProc.h"

#include <stdio.h>


class CRealPreProc : public CSimplePreProc<DREAL>
{
public:
	CRealPreProc(const CHAR *name, const CHAR* id) : CSimplePreProc<DREAL>(name,id)
	{
	}

	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EFeatureType get_feature_type() { return F_DREAL; }
};
#endif
