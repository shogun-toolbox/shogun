#ifndef _CREALFEATURES__H__
#define _CREALFEATURES__H__

#include "preproc/Preproc.h"
#include "features/Features.h"

class CStringFeatures: public CStringFeatures
{
 public:
  CStringFeatures()::CFeatures() { } ;
  ~CStringFeatures() { };
  
  virtual EType get_feature_type() { return STRING ; } ;
  
  /** get feature vector for sample num
      from the matrix as it is if matrix is
      initialized, else return
      preprocessed compute_feature_vector  
      @param num index of feature vector
      @param len length is returned by reference
  */
  char* get_feature_vector(int num, int& len, bool& free);
  bool free_feature_vector(char* feat_vec, bool free);
  
protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual char* compute_feature_vector(int num, int& len)=0;


};
#endif
