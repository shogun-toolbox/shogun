#ifdef SVMMPI
#ifndef _SVMMPI_H___
#define _SVMMPI_H___

/*#ifdef SVMMPI*/

#include "svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "svm_mpi/matrix.h"
#include "features/RealFeatures.h"
#include "svm_mpi/mpi_base.h"

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)

class CSVMMPI: public CSVM, public CMPIBase
{
 public:
  CSVMMPI();
  virtual ~CSVMMPI();
  
  virtual bool svm_train(CFeatures* train);
  virtual REAL* svm_test(CFeatures* test, CFeatures* train);
  virtual bool load(FILE* svm_file);
  virtual bool save(FILE* svm_file);
  
 protected:
  void svm_mpi_optimize(int *labels, int num_examples, CRealFeatures * train) ;

 protected:
  double svm_b, *svm_w ;
} ;
#endif
/*#endif // SVMMPI*/

#endif // _SVMMPI_H__
#endif
