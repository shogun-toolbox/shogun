#ifdef USE_SVMMPI
#ifndef _ONECLASSMPI_H___
#define _ONECLASSMPI_H___

#include "svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "svm_mpi/matrix.h"
#include "features/RealFeatures.h"
#include "svm_mpi/mpi_base.h"
#include "lib/config.h"

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)

class COneClassMPI: public CSVM, public CMPIBase
{
 public:
  COneClassMPI();
  virtual ~COneClassMPI();
  
  virtual bool svm_train(CFeatures* train);
  virtual REAL* svm_test(CFeatures* test, CFeatures* train);
  virtual bool load(FILE* svm_file);
  virtual bool save(FILE* svm_file);
  
 protected:
  void svm_mpi_optimize(int num_examples, CRealFeatures * train) ;

 protected:
  double svm_b, *svm_w, nu ;
} ;
#endif

#endif // _ONECLASSMPI_H__
#endif
