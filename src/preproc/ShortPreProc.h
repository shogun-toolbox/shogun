#ifndef _CSHORTPREPROC__H__
#define _CSHORTPREPROC__H__

#include "features/Features.h"
#include "preproc/SimplePreProc.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CShortPreProc : public CSimplePreProc<SHORT>
{
public:
	CShortPreProc(const CHAR *name, const CHAR* id);
	virtual ~CShortPreProc();

	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EFeatureType get_feature_type() { return F_SHORT; }
};
#endif
