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
