#ifndef _SVMMPI_H___
#define _SVMMPI_H___

/*#ifdef SVMMPI*/

#include "svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "svm_mpi/matrix.h"

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)

class CSVMMPI: public CSVM
{
 public:
  CSVMMPI(int argc, const char **argv);
  virtual ~CSVMMPI();
  
  virtual bool svm_train(CFeatures* train);
  virtual REAL* svm_test(CFeatures* test, CFeatures* train);
  virtual bool load_svm(FILE* svm_file);
  virtual bool save_svm(FILE* svm_file);
  
 protected:
  void svm_mpi_init(int argc, const char **argv); 
  unsigned svm_mpi_broadcast_Z_size(int num_cols, int num_rows, unsigned &m_last) ;
  void svm_mpi_set_Z_block(double * block, int num_cols, int start_idx, int rank) ; 
  void svm_mpi_optimize(int *labels, int num_examples) ;
  void svm_mpi_destroy(void) ;

 protected:
  unsigned m_full, m_last, m_prime;
  unsigned my_rank, num_nodes, num_rows;
  CMatrix<double> Z ;
  
} ;
#endif
/*#endif // SVMMPI*/

#endif // _SVMMPI_H__
