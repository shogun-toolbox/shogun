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

	/** load features from file
	 * fname - filename
	 */

	CFeatures(char* fname);

	virtual ~CFeatures();

	/** return feature type with which objects derived 
	    from CFeatures can deal
	*/
	virtual EFeatureType get_feature_type()=0;

	/** return feature class
	    like Sparse,Simple,...
	*/
	virtual EFeatureClass get_feature_class()=0;
		
	/// set preprocessor
	virtual int add_preproc(CPreProc* p);
	
	/// delete preprocessor from list
	/// caller has to clean up returned preproc
	virtual CPreProc* del_preproc(int num);

	/// get specified preprocessor
	CPreProc* get_preproc(int num);
	
	/// set applied flag for preprocessor
	inline void set_preprocessed(int num) { preprocessed[num]=true; }

	/// get whether specified preprocessor was already applied
	inline bool is_preprocessed(int num) { return preprocessed[num]; }

	/// get number of preprocessors
	inline int get_num_preproc() { return num_preproc; };

	/// get cache size
	inline long get_cache_size() { return cache_size; };

	/// return the number of samples
	virtual long get_num_vectors()=0 ;

	virtual CFeatures* duplicate() const=0 ;

	//in case there is a feature matrix allow for reshaping
	virtual bool reshape(int num_features, int num_vectors) { return false; }

	/** return size (in bytes) a single element (e.g. sizeof(float))
	    requires
	*/
	virtual int get_size()=0;

	virtual bool load(char* fname);
	virtual bool save(char* fname);
	
private:
	/// size of cache in MB
	long cache_size;

	/// list of preprocessors
	CPreProc** preproc;

	/// number of preprocs in list
	int num_preproc;

	/// i'th entry is true if features were already preprocessed with preproc i
	bool* preprocessed;
};
#endif
