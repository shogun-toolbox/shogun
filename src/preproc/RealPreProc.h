#ifndef _CREALPREPROC__H__
#define _CREALPREPROC__H__

#include "lib/common.h"
#include "preproc/SimplePreProc.h"

#include <stdio.h>


class CRealPreProc : public CSimplePreProc<REAL>
{
public:
	CRealPreProc(const CHAR *name, const CHAR* id) : CSimplePreProc<REAL>(name,id)
	{
	}

	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EFeatureType get_feature_type() { return F_REAL; }
};
#endif
