#ifdef SVMMPI
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
#include "intpoint.h"
#include "intpoint_mpi.h"
#include "mpi_base.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include "preproc/PCACut.h"

CBlockCache<double> CMPIBase::bcache_d;
CBlockCache<int> CMPIBase::bcache_i;
unsigned CMPIBase::my_rank, CMPIBase::num_nodes;

extern "C" 
void my_delete(void* ptr)
{
  delete[] ptr ;
} ;

extern "C"
void donothing(void *)
{
  /* do nothing */
}

CMPIBase::CMPIBase()
{
 /* Block caches */
  bcache_d.AddCacheSize(1);
  matrix_set_cache_mgr<double>(&bcache_d);
  matrix_set_cache_mgr<int>(&bcache_i);

  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  
} ;

CMPIBase::~CMPIBase()
{
} ;

template <class I, class T>
void run_non_root_2mpi(const unsigned my_rank, const unsigned num_nodes);

void CMPIBase::svm_mpi_init(int argc, const CHAR **argv)
{ 
  int my_rank, num_nodes ;
  CIO::message("Initializing MPI\n") ;
  MPI_Init(&argc, (CHAR***)&argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);    
  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);

  CIO::message(stderr, "my_rank=%i\nnum_nodes=%i\n", my_rank, num_nodes) ;
  
  if (my_rank) {
    bcache_d.AddCacheSize(1);
    bcache_d.AddCacheSize(100);
    matrix_set_cache_mgr<double>(&bcache_d);
    matrix_set_cache_mgr<int>(&bcache_i);

    run_non_root_2mpi<double,double>(my_rank, num_nodes);
    MPI_Finalize();
    exit(0);
  }
}

unsigned CMPIBase::svm_mpi_broadcast_Z_size(int num_cols, int num_rows_, unsigned &m_last_)
{
  unsigned num_nodes;
  num_rows=num_rows_ ;
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  m_prime=distribute_dimensions(num_cols, num_nodes, num_rows_, num_rows_, &m_last) ;

  m_last_=m_last ;
  m_full = num_cols ;
  Z.Resize(num_rows_, m_prime) ;
  return m_prime ;
} ;

void CMPIBase::svm_mpi_set_Z_block(double * block, int num_cols, int start_idx, int rank) 
{
  //CIO::message("z_set %i %i\n", start_idx, rank) ;
  if (rank)
    {
      // CIO::message("z_client %i %i\n",start_idx, rank) ;
      send_z_columns_double(MPI_COMM_WORLD, block, start_idx,
			    num_cols, num_rows,
			    rank, true);
    }
  else
    {
      //CIO::message("z_block server %i %i %i %i %ld\n",start_idx, rank, num_cols, num_rows, block) ;
      CMatrix<double> tmp(num_rows, num_cols, block, false, donothing);
      Z(colon(0,num_rows), colon(start_idx, start_idx+num_cols-1)) = tmp;
    } ;
} ;

void CMPIBase::svm_mpi_destroy(void)
{
  /* Root node only */
  bcast_req(MPI_COMM_WORLD, 0, REQ_QUIT);
  //bcast_req(MPI_COMM_WORLD, 0, REQ_FINALIZE);
  MPI_Finalize();
}

#endif
#endif
