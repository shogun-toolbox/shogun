#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "lib/lapack.h"
#include "lib/common.h"
#include "preproc/PreProc.h"
#include "stdio.h"

class CPreProc;

class CFeatures
{
public:
	/** Features can 
	 * just be REALs, SHORT
	 * or STRINGs, FILES, or...
	 *
	 * size - cache size
	*/
	CFeatures(long size);

	// copy constructor
	CFeatures(const CFeatures& orig);

	virtual ~CFeatures();

	/** return feature type with which objects derived 
	    from CFeatures can deal
	*/
	virtual EType get_feature_type()=0;
		
	/// set preprocessor
	virtual int add_preproc(CPreProc* p);
	
	/// delete preprocessor from list
	/// caller has to clean up returned preproc
	virtual CPreProc* del_preproc(int num);

	/// get specified preprocessor
	CPreProc* get_preproc(int num);
	
	/// get number of preprocessors
	inline int get_num_preproc() { return num_preproc; };

	/// get cache size
	inline long get_cache_size() { return cache_size; };

	/// preprocess the feature_matrix
	virtual bool preproc_feature_matrix(bool force_preprocessing=false)=0;

	/// set/get the labels
	virtual bool set_label(long idx, int label) { return false ; }
	virtual int  get_label(long idx)=0 ;
	
	/// get label vector
	/// caller has to clean up
	int* get_labels(long &len) ;

	/// return the number of samples
	virtual long get_num_vectors()=0 ;

	virtual CFeatures* duplicate() const=0 ;

	virtual bool save(FILE* dest)=0;
	
private:
	/// size of cache in MB
	long cache_size;

	/// list of preprocessors
	CPreProc** preproc;

	/// number of preprocs in list
	int num_preproc;

protected:
	/// true if features were already preprocessed
	bool preprocessed;
};
#endif
