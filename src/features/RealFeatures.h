#ifndef _CREALFEATURES__H__
#define _CREALFEATURES__H__

#include "preproc/Preproc.h"
#include "features/Features.h"

class CRealFeatures: public CFeatures
{
 public:
  CRealFeatures()::CFeatures() { } ;
  ~CRealFeatures() { };
  
  virtual EType get_feature_type() { return CFeatures::DOUBLE ; } ;
  
  /** get feature vector for sample num
      from the matrix as it is if matrix is
      initialized, else return
      preprocessed compute_feature_vector  
      @param num index of feature vector
      @param len length is returned by reference
  */
  REAL* get_feature_vector(int num, int& len, bool& free);
  bool free_feature_vector(REAL* feat_vec, bool free);
  
  /// get the pointer to the feature matrix
  /// num_feat,num_vectors are returned by reference
  REAL* get_feature_matrix(int &num_feat, int &num_vec);  
  
  /** set feature matrix
      necessary to set feature_matrix, num_features, num_vectors, where
      num_features is the column offset, and columns are linear in memory
      see below for definition of feature_matrix
  */
  virtual REAL* set_feature_matrix()=0;

protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual REAL* compute_feature_vector(int num, int& len)=0;

  REAL* feature_matrix;
  
  /// number of features in cache
  int num_features;
  
  /// number of vectors in cache
  int num_vectors;
};
#endif
