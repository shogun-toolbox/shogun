#ifndef _CPREPROC__H__
#define _CPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CPreProc
{
public:
	CPreProc(const char* name, const char* id);
	virtual ~CPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;
	
	/// initialize preprocessor from file
	virtual bool load_init_data(FILE* src)=0;
	/// save init-data (like transforamtion matrices etc) to file
	virtual bool save_init_data(FILE* dst)=0;

	/// cleanup
	virtual void cleanup()=0;
	
	/** return feature type with which objects derived 
	from CPreProc can deal */
	virtual EType get_feature_type()=0;

	/// return the name of the preprocessor
	const char * get_name() const { return preproc_name; }
	
	/// return a FOUR letter id of the preprocessor
	const char * get_id() const { return preproc_id; }

 protected:
	const char* preproc_name ;
	const char* preproc_id;
};
#endif
