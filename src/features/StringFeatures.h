#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include "preproc/PreProc.h"
#include "features/Features.h"

// StringFeatures do not support PREPROCS
class CStringFeatures: public CFeatures
{
	struct T_STRING
	{
		CHAR* string;
		int length;
	};

	public:
		CStringFeatures();
		CStringFeatures(const CStringFeatures & orig);

		virtual ~CStringFeatures();

		virtual EType get_feature_type() { return F_STRING ; } ;

		/** get feature vector for sample num
		  @param num index of feature vector
		  @param len length is returned by reference
		  */
		CHAR* get_feature_vector(long num, long& len);


		// return false as not available for strings
		virtual bool preproc_feature_matrix(bool force_preprocessing=false) { return false ; }
		virtual long get_num_vectors()=0;

		virtual CFeatures* duplicate() const=0 ;

		virtual bool load(FILE* dest)=0;
		virtual bool save(FILE* dest)=0;

	protected:
		/// number of string vectors
		long num_vectors;

		//this contains the array of features.
		T_STRING* features;
};
#endif
