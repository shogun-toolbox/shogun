#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "lib/lapack.h"
#include "preproc/PreProc.h"

class CPreProc;

class CFeatures
{
public:
	/** Features can 
	 * just be REALs, SHORT
	 * or STRINGs or...
	*/
	enum EType
	{
		F_REAL,
		F_SHORT,
		F_STRING
	};

	CFeatures();
	virtual ~CFeatures();

	/** return feature type with which objects derived 
	    from CFeatures can deal
	*/
	virtual EType get_feature_type()=0;
		
	/// set preprocessor
	virtual void set_preproc(CPreProc* p);

	/// get current preprocessor
	CPreProc* get_preproc();

	/// preprocess the feature_matrix
	bool preproc_feature_matrix();

	/// set/get the labels
	virtual bool set_label(int idx, int label) { return false ; }
	virtual int  get_label(int idx)=0 ;
	
	/// get label vector
	/// caller has to clean up
	int* get_labels(int idx, int &len) ;

	/// return the number of samples
	virtual int get_number_of_examples()=0 ;
	
protected:
	/// Preprocessor
	CPreProc* preproc;
};

#endif
