#ifndef _CSHORTFEATURES__H__
#define _CSHORTFEATURES__H__

#include "preproc/PreProc.h"
#include "features/Features.h"

class CShortFeatures: public CFeatures
{
 public:
  CShortFeatures(long size) ;
  virtual ~CShortFeatures() ;
  
  virtual EType get_feature_type() { return F_SHORT ; } ;
  
  /** get feature vector for sample num
      from the matrix as it is if matrix is
      initialized, else return
      preprocessed compute_feature_vector  
      @param num index of feature vector
      @param len length is returned by reference
  */
  short int* get_feature_vector(long num, long& len, bool& free);
  void free_feature_vector(short int* feat_vec, bool free);
  
  /// get the pointer to the feature matrix
  /// num_feat,num_vectors are returned by reference
  short int* get_feature_matrix(long &num_feat, long &num_vec);  
  
  /** set feature matrix
      necessary to set feature_matrix, num_features, num_vectors, where
      num_features is the column offset, and columns are linear in memory
      see below for definition of feature_matrix
  */
  virtual short* set_feature_matrix()=0;

  virtual bool preproc_feature_matrix();

  int get_num_features() { return num_features ; } ;

protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual void compute_feature_vector(long num, short int* feat)=0;

  /// number of features in cache
  long num_features;
  
  /// number of vectors in cache
  long num_vectors;

  /** chunk of memory for all the feature_vectors	
      it is aligned like 0...num_features-1 for vec0
      0...num_features-1 for vec1 and so on up to vecnum_vectors-1
  */
  short int* feature_matrix;
};
#endif
