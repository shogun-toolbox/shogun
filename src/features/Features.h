#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "preproc/PreProc.h"

class CFeatures
{

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
	
protected:
	/// compute feature vector for sample num
	/// len is returned by reference
	virtual REAL* compute_feature_vector(int num, int& len)=0;

	/// Preprocessor
	CPreProc* preproc;
};
#endif
