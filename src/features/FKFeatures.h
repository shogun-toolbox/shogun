/*#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "preproc/Preproc.h"

class CFeatures
{
    CSVM();
    virtual ~CSVM();


    /// get feature vector for sample num
    inline const REAL* get_feature_vector(int num)
    { 
	if (num<num_vectors)
	    return feature_cache[num*num_features];
	else
	    return get_feature_vector_comp(int num);
    }

protected:
    virtual const REAL* get_feature_vector_comp(int num)=0;
    
    /// set preprocessor
    inline void set_preproc(CPreProc* p) { preproc=p };
    
    /// set current preprocessor
    inline CPreProc* get_preproc() { return preproc; };

protected:
    CPreProc* preproc;

    /** chunk of memory for all the feature_vectors	
      it is aligned like 0...num_features-1 for vec0
      0...num_features-1 for vec1 and so on up to vecnum_vectors-1
    */
    double* feature_cache;

    /// number of features in cache
    int num_features;

    /// number of vectors in cache
    int num_vectors;
};
#endif
*/
