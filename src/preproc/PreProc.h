#ifndef _CPREPROC__H__
#define _CPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CPreProc
{
public:
	CPreProc();
	virtual ~CPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;
	/// cleanup
	virtual void cleanup()=0;
	/// initialize preprocessor from file
	virtual bool load(FILE* f)=0;
	/// save preprocessor init-data to file
	virtual bool save(FILE* f)=0;
	
	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EType get_feature_type()=0;
};
#endif
