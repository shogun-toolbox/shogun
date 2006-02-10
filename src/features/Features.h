#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "preproc/PreProc.h"
#include <stdio.h>

class CPreProc;
class CFeatures;

class CFeatures
{
public:
	/** Features can 
	 * just be REALs, SHORT
	 * or STRINGs, FILES, or...
	 *
	 * size - cache size
	*/
	CFeatures(LONG size);

	// copy constructor
	CFeatures(const CFeatures& orig);

	/** load features from file
	 * fname - filename
	 */

	CFeatures(CHAR* fname);

	virtual CFeatures* duplicate() const=0 ;

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
	virtual INT add_preproc(CPreProc* p);
	
	/// delete preprocessor from list
	/// caller has to clean up returned preproc
	virtual CPreProc* del_preproc(INT num);

	/// get specified preprocessor
	CPreProc* get_preproc(INT num);
	
	/// set applied flag for preprocessor
	inline void set_preprocessed(INT num) { preprocessed[num]=true; }

	/// get whether specified preprocessor was already applied
	inline bool is_preprocessed(INT num) { return preprocessed[num]; }

	/// get number of preprocessors
	inline INT get_num_preproc() { return num_preproc; };

	/// get cache size
	inline LONG get_cache_size() { return cache_size; };

	/// return the number of examples
	virtual INT get_num_vectors()=0 ;

	//in case there is a feature matrix allow for reshaping
	virtual bool reshape(INT num_features, INT num_vectors) { return false; }

	/** return size (in bytes) a single element (e.g. sizeof(float))
	    requires
	*/
	virtual INT get_size()=0;

	void list_feature_obj();

	virtual bool load(CHAR* fname);
	virtual bool save(CHAR* fname);

	bool check_feature_compatibility(CFeatures* f);
	
private:
	/// size of cache in MB
	LONG cache_size;

	/// list of preprocessors
	CPreProc** preproc;

	/// number of preprocs in list
	INT num_preproc;

	/// i'th entry is true if features were already preprocessed with preproc i
	bool* preprocessed;
};
#endif
