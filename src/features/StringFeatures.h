#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include "preproc/PreProc.h"
#include "features/Features.h"

class CStringFeatures: public CFeatures
{
	public:
		CStringFeatures(long size);
		CStringFeatures(const CStringFeatures & orig);

		virtual ~CStringFeatures();

		virtual EType get_feature_type() { return F_STRING ; } ;

		/** get feature vector for sample num
		  from the matrix as it is if matrix is
		  initialized, else return
		  preprocessed compute_feature_vector  
		  @param num index of feature vector
		  @param len length is returned by reference
		  */
		char* get_feature_vector(int num, int& len, bool& free);
		void free_feature_vector(CHAR* feat_vec, int num, bool free);

		virtual bool preproc_feature_matrix(bool force_preprocessing=false);
		virtual long get_num_vectors()=0 ;
		virtual CFeatures* duplicate() const=0 ;
		virtual bool save(FILE* dest)=0;

	protected:
		/// compute feature vector for sample num
		/// len is returned by reference
		virtual char* compute_feature_vector(int num, int& len)=0;

		CHAR* feature_matrix;
};
#endif
