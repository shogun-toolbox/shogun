#ifdef USE_SVMMPI
#ifndef _MPIBASE_H___
#define _MPIBASE_H___

#include "lib/config.h"
#include "classifier/svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "svm_mpi/matrix.h"
#include "features/RealFeatures.h"

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)

extern "C" void my_delete(void* ptr) ;
extern "C" void donothing(void *) ;

class CMPIBase
{
 public:
  CMPIBase();
  virtual ~CMPIBase();
  
  static void svm_mpi_init(int argc, const CHAR **argv); 
  static void svm_mpi_destroy(void) ;

 protected:
  unsigned svm_mpi_broadcast_Z_size(int num_cols, int num_rows, unsigned &m_last) ;
  void svm_mpi_set_Z_block(double * block, int num_cols, int start_idx, int rank) ; 

 protected:
  static unsigned my_rank, num_nodes;
  static CBlockCache<double> bcache_d;
  static CBlockCache<int> bcache_i;
  unsigned m_last, m_full, m_prime, num_rows ;
  CMatrix<double> Z ;
} ;
#endif

#endif // _MPIBASE_H__

#endif
