#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "RealPreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CPCACut : public CRealPreProc
{
 public:
  CPCACut();
  virtual ~CPCACut();
  
  /// initialize preprocessor from features
  virtual bool init(CFeatures* f);
  /// cleanup
  virtual void cleanup();
  /// initialize preprocessor from file
  virtual bool load(FILE* f);
  /// save preprocessor init-data to file
  virtual bool save(FILE* f);
  
  /// apply preproc on feature matrix
  /// result in feature matrix
  /// return pointer to feature_matrix, i.e. f->get_feature_matrix();
  virtual REAL* apply_to_feature_matrix(CFeatures* f);
  
  /// apply preproc on single feature vector
  /// result in feature matrix
  virtual REAL* apply_to_feature_vector(REAL* f, int &len);

 protected:
  double* T ;
  int num_dim ;
  double *mean ;
};
#endif
