#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "intpoint.h"
#include "intpoint_mpi.h"
#include "mpi_svm.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include "preproc/PCACut.h"

CBlockCache<double> CSVMMPI::bcache_d;
CBlockCache<int> CSVMMPI::bcache_i;

int sign(double a)
{
  if (a>0)
    return 1 ;
  if (a<0)
    return -1 ;
  return 0 ;
} 

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

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
CSVMMPI::CSVMMPI()
  : svm_b(0), svm_w(NULL)
{
 /* Block caches */
  bcache_d.AddCacheSize(1);
  bcache_d.AddCacheSize(100);
  matrix_set_cache_mgr<double>(&bcache_d);
  matrix_set_cache_mgr<int>(&bcache_i);

  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  
  kernel=NULL ;
} ;

CSVMMPI::~CSVMMPI()
{
  delete[] svm_w ;
} ;


bool CSVMMPI::svm_train(CFeatures* train_)
{
  if (train_->get_feature_type()!=F_REAL)
    {
      CIO::message("features do not fit to SVM\n") ;
      return false ;
    } ;
  CRealFeatures *train=(CRealFeatures*)train_ ;
  
  int num_cols=train->get_num_vectors() ;
  num_rows=((CRealFeatures*)train)->get_num_features() ;
  CIO::message("num_rows=%i\n", num_rows) ;
  long dummy ;
  int * labels=train->get_labels(dummy) ;
  assert(dummy==num_cols) ;

  CIO::message("creating big matrix ... ") ;

  bool free ; long len ;
  double * column=NULL ;
  column=train->get_feature_vector(0, len, free);
  train->free_feature_vector(column, free);
  num_rows=len ;

  //CIO::message(" [%ix%i] ", num_cols, len) ;
  //CIO::message(" and saving to file ~/Z_clean.dat") ;

  m_prime=svm_mpi_broadcast_Z_size(num_cols, num_rows, m_last) ;
  //CIO::message("mprime=%i, m_last=%i\n ", m_prime, m_last) ;
  int j=0;
  
  for (j=0; j<num_cols; j++) 
  {
    int rank=floor(((double)j)/m_prime) ;
    int start_idx=j%m_prime ; 

    //CIO::message("mprime=%i, rank=%i\n ", m_prime, rank) ;
    
    if (!(j % (num_cols/10+1)))
      CIO::message("%02d%%.", (int) (100.0*j/num_cols));
    else if (!(j % (num_cols/200+1)))
      CIO::message(".");

    column=train->get_feature_vector(j, len, free);

    REAL *col=new REAL[len] ;
    for (int kk=0; kk<len; kk++)
      col[kk]=column[kk] ; 

    train->free_feature_vector(column, free);

    assert(len==num_rows) ;
    svm_mpi_set_Z_block(col, 1, start_idx, rank) ; 
    delete[] col ;
  } ;
  CIO::message("Done\n") ;

  svm_mpi_optimize(labels, num_cols, train) ; 
  return true; 
}

REAL* CSVMMPI::svm_test(CFeatures* test_, CFeatures*)
{
  CRealFeatures * test=(CRealFeatures*)test_ ;
  long num_test=test->get_num_vectors();
  REAL* output=new REAL[num_test];
  for (long i=0; i<num_test;  i++)
    {
      int onei=1 ;
      double zerod=0, oned=1 ;
      char N='N' ;
      char T='T' ;
      long len ; int length ;
      bool free ;
      double*feature=test->get_feature_vector(i, len, free) ;
      length=len ;

      output[i]=ddot_(&length,feature,&onei, svm_w, &onei) ;
  
      test->free_feature_vector(feature, free) ;
    } ;
  if (1) {
    int i ;
    for (i=0; i<10; i++)
      CIO::message("output[%i]=%e\n", i, output[i]-svm_b) ;
  } ;

  return output;  
}

bool CSVMMPI::load(FILE* modelfl)
{
	bool result=false;
	char version_buffer[1024];

	fscanf(modelfl,"MPI\n", version_buffer);
	if(strcmp(version_buffer,"MPI")) {
		perror ("model file does not match MPI SVM");
		exit (1); 
	}

	fscanf(modelfl,"%ld%*[^\n]\n", &num_rows);

	delete[] svm_w;
	svm_w = new double[num_rows];
	CIO::message("loading w of size %ld\n",num_rows);

	for (int i=0; i<num_rows; i++)
		fscanf(modelfl,"%lf%*[^\n]\n", svm_w[i]);

	CIO::message("done\n");
	result=true;
	return result;
} 

bool CSVMMPI::save(FILE* modelfl)
{
  CIO::message("Writing model file...");
  fprintf(modelfl,"MPI\n");
  fprintf(modelfl,"%ld # length of w\n", num_rows);
  fprintf(modelfl,"%+10.16e # threshold b \n",svm_b);
  
  for(int i=0;i<num_rows;i++)
    fprintf(modelfl,"%+10.16e\n", svm_w[i]);
  
  CIO::message("done\n");
  return true ;
}



template <class I, class T>
void run_non_root_2mpi(const unsigned my_rank, const unsigned num_nodes);


void CSVMMPI::svm_mpi_init(int argc, const char **argv)
{ 
  int my_rank, num_nodes ;
  CIO::message("Initializing MPI\n") ;
  MPI_Init(&argc, (char***)&argv);
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

unsigned CSVMMPI::svm_mpi_broadcast_Z_size(int num_cols, int num_rows_, unsigned &m_last_)
{
  unsigned num_nodes;
  num_rows=num_rows_ ;
  MPI_Comm_size(MPI_COMM_WORLD, (int *)&num_nodes);
  m_prime=distribute_dimensions(num_cols, num_nodes, num_rows_, num_rows_, &m_last) ;
  //num_nodes=1 ;
  //m_prime=num_cols ;
  //m_last=num_cols ;

  m_last_=m_last ;
  m_full = num_cols ;
  Z.Resize(num_rows_, m_prime) ;
  return m_prime ;
} ;

void CSVMMPI::svm_mpi_set_Z_block(double * block, int num_cols, int start_idx, int rank) 
{
  //CIO::message("z_set %i %i\n", start_idx, rank) ;
  if (rank)
    {
      //CIO::message("z_client %i %i\n",start_idx, rank) ;
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

void CSVMMPI::svm_mpi_optimize(int * labels, int num_examples, CRealFeatures * train) 
{
  double bound=10 ;
  int maxiter=50 ;

  CIO::message("preparing small matrices: num_examples=%i\n",num_examples) ;

  IntpointResources *res = NULL ;
  if (! my_rank)
    res = new IntpointResources[num_nodes];

  double *dc=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    dc[i]=-(double)labels[i] ;
  double *dA=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    dA[i]=1;

  CIO::message("C=%e\n", C) ;
  double *ub=new double[num_examples],*lb=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    {
      if (labels[i]==1)
	{
	  ub[i]=C ; lb[i]=0 ;
	} ;
      if (labels[i]==-1)
	{
	  ub[i]=0 ; lb[i]=-C ;
	}
    } ;

  double zval = 0.0;
  double dr = 1.0;
  CMatrix<double> *b=new CMatrix<double>(1, 1, &zval, false, donothing);
  CMatrix<double> *l=new CMatrix<double>(m_full, 1, lb, false, my_delete);
  CMatrix<double> *u=new CMatrix<double>(m_full, 1, ub, false, my_delete);
  CMatrix<double> *c=new CMatrix<double>(num_examples, 1, dc, false, my_delete);
  CMatrix<double> *r=new CMatrix<double>(1, 1, &zval, false, donothing);
  CMatrix<double> *A=new CMatrix<double>(1, num_examples, dA, false, my_delete);
  
  CIntPointPR optimizer;
  optimizer.SetBound(10);
  optimizer.SetMaxIterations(maxiter);

  unsigned my_rank=0 ;
  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&my_rank);
  const char * how ;

  CMatrix<double> *primal=new CMatrix<double>(), 
    *dual=new CMatrix<double>() ;

  CIO::message("starting optimizer\n") ;

  #ifdef DEBUG
  //#define HOME "/opt/home/raetsch/"
  #define HOME "/home/104/gxr104/short/"
  {
    CIO::message("saving Z matrix to ~/Z.dat (%ix%i)\n",Z.GetNumRows(),Z.GetNumColumns()) ;    
    double* d=Z.GetDataPointer() ;
    FILE* f=fopen(HOME "Z.dat","w+") ;
    fwrite(d, sizeof(double), Z.GetNumRows()*Z.GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving A matrix to ~/A.dat\n") ;    
    double* d=A->GetDataPointer() ;
    FILE* f=fopen(HOME "A.dat","w+") ;
    fwrite(d, sizeof(double), A->GetNumRows()*A->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving c vector to ~/c.dat\n") ;    
    double* d=c->GetDataPointer() ;
    FILE* f=fopen(HOME "c.dat","w+") ;
    fwrite(d, sizeof(double), c->GetNumRows()*c->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving l vector to ~/l.dat\n") ;    
    double* d=l->GetDataPointer() ;
    FILE* f=fopen(HOME "l.dat","w+") ;
    fwrite(d, sizeof(double), l->GetNumRows()*l->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving u vector to ~/u.dat\n") ;    
    double* d=u->GetDataPointer() ;
    FILE* f=fopen(HOME "u.dat","w+") ;
    fwrite(d, sizeof(double), u->GetNumRows()*u->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving r vector to ~/r.dat\n") ;    
    double* d=r->GetDataPointer() ;
    FILE* f=fopen(HOME "r.dat","w+") ;
    fwrite(d, sizeof(double), r->GetNumRows()*r->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving b vector to ~/b.dat\n") ;    
    double* d=b->GetDataPointer() ;
    FILE* f=fopen(HOME "b.dat","w+") ;
    fwrite(d, sizeof(double), b->GetNumRows()*b->GetNumColumns(), f) ;
    fclose(f) ;
  } 
#endif 
  optimize_smw2mpi_core<double>(optimizer, *c, Z, *A, *b, *l, *u,
				*r, m_prime, m_last, my_rank,
				num_nodes, res, 1, NULL/*"/short/x46/tmp/x.mat"*/,
				NULL /*"/short/x46/tmp/x.mat"*/,
				*primal, *dual, &how);
  double *prim = primal->GetDataPointer();
  double *dua = dual->GetDataPointer();
  double *Zd = Z.GetDataPointer();

#ifdef DEBUG
  {
    CIO::message("saving prim vector to ~/primal.dat\n") ;    
    double* d=primal->GetDataPointer() ;
    FILE* f=fopen(HOME "primal.dat","w+") ;
    fwrite(d, sizeof(double), primal->GetNumRows()*primal->GetNumColumns(), f) ;
    fclose(f) ;
  } 
  {
    CIO::message("saving dual vector to ~/dual.dat\n") ;    
    double* d=dual->GetDataPointer() ;
    FILE* f=fopen(HOME "dual.dat","w+") ;
    fwrite(d, sizeof(double), dual->GetNumRows()*dual->GetNumColumns(), f) ;
    fclose(f) ;
  }
#endif

  delete[] svm_w ;
  svm_w=new REAL[num_rows] ;
  svm_b=*dua ;

  CIO::message("num_rows=%i\n",num_rows) ;

  int onei=1 ;
  double zerod=0, oned=1 ;
  char N='N' ;
  int num_rows_=num_rows ;

  //dgemv_(&N, &num_rows_, &num_examples, &oned, Zd, &num_rows_, prim, &onei, &zerod, svm_w, &onei) ;
  {
    int i;
    for (i=0; i<num_rows; i++)
      svm_w[i]=0.0 ;
    for (i=0; i<num_examples; i++)
      {
	long len ; bool free ; double oned=1 ; int onei=0 ; int length ; int j;
	double * feat=train->get_feature_vector(i, len, free) ;
	assert(num_rows==len) ;
	length=len ;
	if (i==0)
	  {
	    for (j=0; j<num_rows; j++)
	      CIO::message("feat[1][%i]=%e\n", j, feat[j]) ;
	  } ;
	for (j=0; j<num_rows; j++)
	  svm_w[j]+= prim[i]*feat[j];
	
	    //daxpy_(&length, &prim[i], feat, &onei, svm_w, &onei) ;
	train->free_feature_vector(feat, free) ;
      } 
  } 
#ifdef DEBUG      
  {
    CIO::message("saving svm_w vector to ~/w.dat\n") ;    
    FILE* f=fopen(HOME "w.dat","w+") ;
    fwrite(svm_w, sizeof(double), num_rows, f) ;
    fclose(f) ;
  } 
  
  if (0) {
    int i; 
    for (i=0; i<num_examples; i++)
      CIO::message("alpha[%i]=%e\n", i, prim[i]) ;
    //    for (i=2499; i<2510; i++)
    //      CIO::message("alpha[%i]=%e\n", i, prim[i]) ;
  } ;
  {
    for (int i=0; i<num_rows; i++)
      CIO::message("w[%i]=%e\n", i, svm_w[i]) ;
    CIO::message("b=%e\n", svm_b) ;
  } ;
  
  REAL* out=svm_test(train, train) ;
  
  if (1) {
    int i ;
    int num_errors=0 ;
    for (i=0; i<num_examples; i++)
      if (sign(out[i])!=labels[i])
	num_errors++ ;
    CIO::message("Training errors: %1.2f (%i)\n", (double)num_errors/num_examples, num_errors) ;
  } ;

  if (0) {
    int i ;
    for (i=0; i<num_examples; i++)
      CIO::message("out[%i]=%e\n", i, out[i]-svm_b) ;
    //    for (i=2499; i<2510; i++)
    //    CIO::message("out[%i]=%e\n", i, out[i]-*dua) ;
  } ;
#endif
  delete c; 
  delete A; 
  delete b; 
  delete l; 
  delete u; 
  delete r;
  delete dual ;
  delete primal ;
} ;


void CSVMMPI::svm_mpi_destroy(void)
{
  /* Root node only */
  bcast_req(MPI_COMM_WORLD, 0, REQ_QUIT);
  MPI_Finalize();
}
#endif
