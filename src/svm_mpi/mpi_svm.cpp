#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "intpoint.h"
#include "intpoint_mpi.h"
#include "mpi_svm.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

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

static double one=1.0 ;

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
CSVMMPI::CSVMMPI(int argc, const char **argv)
  //  : Z(1,1,&one,false,donothing)
{
  //  svm_mpi_init(argc, argv) ;
  kernel=NULL ;
} ;

CSVMMPI::~CSVMMPI()
{
  svm_mpi_destroy() ;
} ;


bool CSVMMPI::svm_train(CFeatures* train_)
{
  if (train_->get_feature_type()!=F_REAL)
    {
      CIO::message("features do not fit to SVM\n") ;
      return false ;
    } ;
  CRealFeatures *train=(CRealFeatures*)train_ ;
  
  int num_cols=train->get_number_of_examples() ;
  num_rows=((CRealFeatures*)train)->get_num_features() ;
  CIO::message("num_rows=%i\n", num_rows) ;
  long dummy ;
  int * labels=train->get_labels(dummy) ;
  assert(dummy==num_cols) ;
  
  CIO::message("creating big matrix\n") ;

  m_prime=svm_mpi_broadcast_Z_size(num_cols, num_rows, m_last) ;
  int j=0;
  
  double * column=NULL ;
  for (j=0; j<num_cols; j++) 
  {
    int rank=floor(((double)j)/m_prime) ;
    int start_idx=j%m_prime ; 
    bool free ; long len ;

    //CIO::message("setting vector: %i %i (%i,%i)\n",start_idx, rank, j, m_prime) ;
    column=train->get_feature_vector(j, len, free);
    assert(len==num_rows) ;
    svm_mpi_set_Z_block(column, 1, start_idx, rank) ; 
    train->free_feature_vector(column, free);
  } ;
  
  svm_mpi_optimize(labels, num_cols) ; 
  return true; 
}

REAL* CSVMMPI::svm_test(CFeatures* test, CFeatures* train)
{
}

bool CSVMMPI::load_svm(FILE* svm_file)
{
} 

bool CSVMMPI::save_svm(FILE* svm_file)
{
}



template <class I, class T>
void run_non_root_2mpi(const unsigned my_rank, const unsigned num_nodes);


void CSVMMPI::svm_mpi_init(int argc, const char **argv)
{ 
  MPI_Init(&argc, (char***)&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  fprintf(stderr, "my_rank=%i\nnum_nodes=%i\n", my_rank, num_nodes) ;

  if (my_rank) {
    run_non_root_2mpi<double,double>(my_rank, num_nodes);
    //run_non_root_2mpi(my_rank, num_nodes);
    MPI_Finalize();
    exit(0);
  }
    
}

unsigned CSVMMPI::svm_mpi_broadcast_Z_size(int num_cols, int num_rows_, unsigned &m_last_)
{
  unsigned num_nodes;
  num_rows=num_rows_ ;
  //MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  //m_prime=distribute_dimensions(num_cols, num_nodes, num_rows_, num_rows_, &m_last) ;
  num_nodes=1 ;
  m_prime=num_cols ;
  m_last=num_cols ;

  m_last_=m_last ;
  m_full = num_cols ;
  Z.Resize(num_rows_, m_prime) ;
  return m_full ;
} ;


void CSVMMPI::svm_mpi_set_Z_block(double * block, int num_cols, int start_idx, int rank) 
{
  if (rank)
    {
      CIO::message("z_client %i %i\n",start_idx, rank) ;
      send_z_columns_double(MPI_COMM_WORLD, block, start_idx,
			    num_cols, num_rows,
			    rank, true);
    }
  else
    {
      //CIO::message("z_block server %i %i %i %i %ld\n",start_idx, rank, num_cols, num_rows, block) ;
      CMatrix<double> tmp(num_rows, num_cols, block, false, donothing);
      Z(colon(), colon(start_idx, start_idx+num_cols-1)) = tmp;
    } ;
} ;

void CSVMMPI::svm_mpi_optimize(int * labels, int num_examples) 
{
  double bound=10 ;
  int maxiter=50 ;

  CIO::message("preparing small matrices\n") ;
  IntpointResources *res = NULL ;
  if (! my_rank)
    res = new IntpointResources[num_nodes];

  double *dlabels=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    dlabels[i]=(double)labels[i] ;

  double zval = 0.0;
  double dr = 1.0;
  CMatrix<double> b(1, 1, &zval);
  CMatrix<double> l(zeros<double>(m_full, 1));
  CMatrix<double> u(ones<double>(m_full, 1));
  u = C * u;
  CMatrix<double> c(-ones<double>(m_full, 1));
  CMatrix<double> r(1, 1, &dr);
  CMatrix<double> A(1, num_examples, dlabels, false,my_delete);
  
  CIntPointPR optimizer;
  optimizer.SetBound(10);
  optimizer.SetMaxIterations(maxiter);

  unsigned my_rank=0 ;
  //MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  const char * how ;

  CMatrix<double> primal, dual ;

  CIO::message("starting optimizer\n") ;

  optimize_smw2mpi_core<double>(optimizer, c, Z, A, b, l, u,
				r, m_prime, m_last, my_rank,
				num_nodes, res, primal,
				dual, &how);
  
  double *prim = primal.GetDataPointer();
  double *dua = dual.GetDataPointer();
} ;


void CSVMMPI::svm_mpi_destroy(void)
{
  /* Root node only */
  bcast_req(MPI_COMM_WORLD, 0, REQ_QUIT);
  MPI_Finalize();
}
#endif
