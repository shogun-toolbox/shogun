#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "RealPreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CPCACut : public CRealPreProc
{
 public:
  CPCACut(int do_whitening=0, double thresh=1e-6);
  virtual ~CPCACut();
  
  /// initialize preprocessor from features
  virtual bool init(CFeatures* f);
  /// initialize preprocessor from file
  virtual bool load_init_data(FILE* src);
  /// save init-data (like transforamtion matrices etc) to file
  virtual bool save_init_data(FILE* dst);
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
  double* T ;
  int num_dim;
  int num_old_dim;
  double *mean ;

  /// true when already initialized
  bool initialized;

  int do_whitening;
  double thresh ;
};
#endif
