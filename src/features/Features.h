#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "preproc/PreProc.h"

class CFeatures
{
 public:
	/** Features can be DOUBLEs and STRINGs
	*/
	enum EType
	{
		REAL,
		SHORT,
		STRING
	};

	CFeatures();
	virtual ~CFeatures();

	/** return feature type with which objects derived 
	    from CFeatures can deal
	*/
	virtual EType get_feature_type()=0;
		
	/// set preprocessor
	virtual void set_preproc(CPreProc* p);

	/// set current preprocessor
	CPreProc* get_preproc();

	/// Preprocess the feature feature_matrix
	bool preproc_feature_matrix();

	virtual bool set_label(int idx, int label) { return false ; }
	virtual int get_label(int idx)=0 ;
	
protected:
	/// Preprocessor
	CPreProc* preproc;
};

#endif
