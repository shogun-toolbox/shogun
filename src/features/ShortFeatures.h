#ifndef _CREALFEATURES__H__
#define _CREALFEATURES__H__

#include "preproc/Preproc.h"
#include "features/Features.h"

class CShortFeatures: public CFeatures
{
 public:
  CShortFeatures()::CFeatures() { } ;
  ~CShortFeatures() { };
  
  virtual EType get_feature_type() { return CFeatures::SHORT ; } ;
  
  /** get feature vector for sample num
      from the matrix as it is if matrix is
      initialized, else return
      preprocessed compute_feature_vector  
      @param num index of feature vector
      @param len length is returned by reference
  */
  short int* get_feature_vector(int num, int& len, bool& free);
  bool free_feature_vector(short int* feat_vec, bool free);
  
protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual short int* compute_feature_vector(int num, int& len)=0;

  /** chunk of memory for all the feature_vectors	
      it is aligned like 0...num_features-1 for vec0
      0...num_features-1 for vec1 and so on up to vecnum_vectors-1
  */
  short int* feature_matrix;

  /// number of features in cache
  int num_features;
  
  /// number of vectors in cache
  int num_vectors;

};
#endif
