#ifdef SVMMPI
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
#include "intpoint.h"
#include "intpoint_mpi.h"
#include "mpi_oneclass.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include "preproc/PCACut.h"

static int sign(double a)
{
  if (a>0)
    return 1 ;
  if (a<0)
    return -1 ;
  return 0 ;
} 

COneClassMPI::COneClassMPI()
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

COneClassMPI::~COneClassMPI()
{
  delete[] svm_w ;
} ;


bool COneClassMPI::svm_train(CFeatures* train_)
{
  if (train_->get_feature_type()!=F_REAL)
    {
      CIO::message("features do not fit to SVM\n") ;
      return false ;
    } ;
  CRealFeatures *train=(CRealFeatures*)train_ ;
  
  int num_cols=train->get_num_vectors() ;
  num_rows=((CRealFeatures*)train)->get_num_features() ;
  CIO::message("num_features=%i\n", num_rows) ;

  LONG dummy ;
  int * labels=train->get_labels(dummy) ;
  assert(dummy==num_cols) ;
  CIO::message("num_examples=%i\n", dummy) ;

  int num_positive=0,i;
  for (int i=0; i<dummy; i++)
    if (labels[i]>0) num_positive++ ;
  CIO::message("num_pos_examples=%i\n", num_positive) ;

  CIO::message("creating big matrix ... ") ;

  bool free ; LONG len ;
  double * column=NULL ;
  column=train->get_feature_vector(0, len, free);
  train->free_feature_vector(column, 0, free);
  assert(num_rows==len) ;

  m_prime=svm_mpi_broadcast_Z_size(num_positive, num_rows, m_last) ;
  int j, idx_positive=0 ;
  for (j=0; j<num_cols; j++) 
    if (labels[j]>0)
      {
	int rank=floor(((double)idx_positive)/m_prime) ;
	int start_idx=idx_positive%m_prime ; 
	
	if (!(idx_positive % (num_positive/10+1)))
	  CIO::message("%02d%%.", (int) (100.0*idx_positive/num_positive));
	else if (!(idx_positive % (num_positive/200+1)))
	  CIO::message(".");

	idx_positive++ ;
	
	column=train->get_feature_vector(j, len, free);
	
	REAL *col=new REAL[len] ;
	for (int kk=0; kk<len; kk++)
	  col[kk]=column[kk] ; 
	
	train->free_feature_vector(column, j, free);
	
	assert(len==num_rows) ;
	svm_mpi_set_Z_block(col, 1, start_idx, rank) ; 
	delete[] col ;
      } ;
  CIO::message("Done\n") ;

  svm_mpi_optimize(num_positive, train) ; 

  return true; 
}

REAL* COneClassMPI::svm_test(CFeatures* test_, CFeatures*)
{
	CRealFeatures * test=(CRealFeatures*)test_ ;
	LONG num_test=test->get_num_vectors();
	CIO::message("testing.\n");
	REAL* output=new REAL[num_test];
	for (LONG i=0; i<num_test;  i++)
	{
		int onei=1 ;
		double zerod=0, oned=1 ;
		CHAR N='N' ;
		CHAR T='T' ;
		LONG len ; int length ;
		bool free ;
		double*feature=test->get_feature_vector(i, len, free) ;
		length=len ;
		/*for (int j=0; j<length; j++)
		  {
		    fprintf(stderr,"%i: %e*%e=%e\n", j,feature[j],svm_w[j]) ;
		    } ;*/

		output[i]=ddot_(&length,feature,&onei, svm_w, &onei)-svm_b ;

		test->free_feature_vector(feature, i, free) ;
	}

	return output;  
}

bool COneClassMPI::load(FILE* modelfl)
{
	bool result=false;
	CHAR version_buffer[1024];

	fscanf(modelfl,"%s\n", version_buffer);
	CIO::message("detected:%s\n", version_buffer);
	if(strcmp(version_buffer,"MPI")) {
		perror ("model file does not match MPI SVM");
		exit (1); 
	}

	fscanf(modelfl,"%ld%*[^\n]\n", &num_rows);
	fscanf(modelfl,"%lf%*[^\n]\n", &svm_b);
	CIO::message("svm_b:%f\n", svm_b);

	delete[] svm_w;
	svm_w = new double[num_rows];
	CIO::message("loading w of size %ld\n",num_rows);

	for (int i=0; i<num_rows; i++)
		fscanf(modelfl,"%lf%*[^\n]\n", &svm_w[i]);

	CIO::message("done\n");
	result=true;
	return result;
} 

bool COneClassMPI::save(FILE* modelfl)
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

void COneClassMPI::svm_mpi_optimize(int num_examples, CRealFeatures * train) 
{
  double bound=10 ;
  int maxiter=50 ;

  CIO::message("preparing small matrices: num_examples=%i\n",num_examples) ;

  IntpointResources *res = NULL ;
  if (! my_rank)
    res = new IntpointResources[num_nodes];

  double *dc=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    dc[i]=0 ; 
  double *dA=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    dA[i]=1;

  CIO::message("C=%e\n", C) ;
  double *ub=new double[num_examples],*lb=new double[num_examples] ;
  for (int i=0; i<num_examples; i++)
    {
      ub[i]=1.0/((double)num_examples) ; lb[i]=0 ;
    } ;

  double zval = 0.0; double nuval=C ;
  double dr = 1.0;
  CMatrix<double> *b=new CMatrix<double>(1, 1, &nuval, false, donothing);
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
  const CHAR * how ;

  CMatrix<double> *primal=new CMatrix<double>(), 
    *dual=new CMatrix<double>() ;

  CIO::message("starting optimizer\n") ;

  optimize_smw2mpi_core<double>(optimizer, *c, Z, *A, *b, *l, *u,
				*r, m_prime, m_last, my_rank,
				num_nodes, res, 0, NULL,
				NULL, *primal, *dual, &how);
  double *prim = primal->GetDataPointer();
  double *dua = dual->GetDataPointer();
  double *Zd = Z.GetDataPointer();

  delete[] svm_w ;
  svm_w=new REAL[num_rows] ;
  svm_b=*dua ;

  CIO::message("num_rows=%i\n",num_rows) ;

  int onei=1 ;
  double zerod=0, oned=1 ;
  CHAR N='N' ;
  int num_rows_=num_rows ;

  {
    int i;
    for (i=0; i<num_rows; i++)
      svm_w[i]=0.0 ;
    for (i=0; i<num_examples; i++)
      {
	LONG len ; bool free ; int length, j;
	double * feat=train->get_feature_vector(i, len, free) ;
	assert(num_rows==len) ;
	length=len ;
	for (j=0; j<num_rows; j++)
	  svm_w[j]+= prim[i]*feat[j];
	
	// int double oned=1 ; int onei=0 ; 
        // daxpy_(&length, &prim[i], feat, &onei, svm_w, &onei) ;
	train->free_feature_vector(feat, i, free) ;
      } 
  } 

  delete c; 
  delete A; 
  delete b; 
  delete l; 
  delete u; 
  delete r;
  delete dual ;
  delete primal ;
} ;

#endif
#endif
