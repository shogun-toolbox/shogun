#ifndef _CPRUNE_VAR_SUB_MEAN__H__
#define _CPRUNE_VAR_SUB_MEAN__H__

#include "RealPreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CPruneVarSubMean : public CRealPreProc
{
 public:
  CPruneVarSubMean();
  virtual ~CPruneVarSubMean();
  
  /// initialize preprocessor from features
  virtual bool init(CFeatures* f);
  /// cleanup
  virtual void cleanup();
  
  /// apply preproc on feature matrix
  /// result in feature matrix
  /// return pointer to feature_matrix, i.e. f->get_feature_matrix();
  virtual REAL* apply_to_feature_matrix(CFeatures* f);
  
  /// apply preproc on single feature vector
  /// result in feature matrix
  virtual REAL* apply_to_feature_vector(REAL* f, int &len);

 protected:
  int* idx ;
  int num_idx ;
};
#endif
