#include "TOPFeatures.h"
#include "lib/io.h"
#include <assert.h>

CTOPFeatures::CTOPFeatures(CHMM* p, CHMM* n)
{
  pos=p;
  neg=n;
  num_vectors=get_number_of_examples() ;
  // set_feature_matrix();
}
CTOPFeatures::CTOPFeatures(const CTOPFeatures &orig): 
	CRealFeatures(orig), pos(orig.pos), neg(orig.neg)
{ 
}

CTOPFeatures::~CTOPFeatures()
{
}

void CTOPFeatures::set_models(CHMM* p, CHMM* n)
{
  pos=p ; 
  neg=n ;
  delete[] feature_matrix  ;
  feature_matrix=NULL ;
  //  set_feature_matrix() ;
}

int CTOPFeatures::get_num_features()
{
  CIO::message("pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i]\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M()) ;
  if (pos && neg)
    return 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ;

  return 0 ;
}

int CTOPFeatures::get_label(long idx)
{
  if (pos && pos->get_observations())
    return pos->get_observations()->get_label(idx) ;
  assert(0) ;
  return 0 ;
}

long CTOPFeatures::get_number_of_examples()
{
  if (pos && pos->get_observations())
    return pos->get_observations()->get_DIMENSION();
  
  return 0;
}
  
CFeatures* CTOPFeatures::duplicate() const
{
	return new CTOPFeatures(*this);
}

REAL* CTOPFeatures::compute_feature_vector(long num, long &len)
{
  REAL* featurevector;
  
  //CIO::message("allocating %.2f M for top feature vector cache\n", 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()));
  
  featurevector=new REAL[ 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ];
  
  if (!featurevector)
    return NULL;
  
  compute_feature_vector(featurevector, num, len);

  return featurevector;
}

void CTOPFeatures::compute_feature_vector(REAL* featurevector, long num, long& len)
{
  long i,j,p=0,x=num;
  
  double posx=pos->model_probability(x);
  double negx=neg->model_probability(x);
  
  len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());
  //  CIO::message("len=%i\n",len) ;
  
  featurevector[p++]=(posx-negx);
  
  //first do positive model
  for (i=0; i<pos->get_N(); i++)
    {
      featurevector[p++]=exp(pos->model_derivative_p(i, x)-posx);
      //  CIO::message("pos_p_deriv=%e\n", featurevector[p-1]) ;
      featurevector[p++]=exp(pos->model_derivative_q(i, x)-posx);
      //CIO::message("pos_q_deriv=%e\n", featurevector[p-1]) ;
      
      for (j=0; j<pos->get_N(); j++) {
	featurevector[p++]=exp(pos->model_derivative_a(i, j, x)-posx);
	//CIO::message("pos_a_deriv[%i]=%e\n", j, featurevector[p-1]) ;
      }
      
      for (j=0; j<pos->get_M(); j++) {
	featurevector[p++]=exp(pos->model_derivative_b(i, j, x)-posx);
	//CIO::message("pos_b_deriv[%i]=%e\n", j, featurevector[p-1]) ;
      } 
      
    }
  
  //then do negative
  for (i=0; i<neg->get_N(); i++)
    {
      featurevector[p++]= - exp(neg->model_derivative_p(i, x)-negx);
      //CIO::message("pos_p_deriv=%e\n", featurevector[p-1]) ;
      featurevector[p++]= - exp(neg->model_derivative_q(i, x)-negx);
      //CIO::message("pos_q_deriv=%e\n", featurevector[p-1]) ;
      
      for (j=0; j<neg->get_N(); j++) {
	featurevector[p++]= - exp(neg->model_derivative_a(i, j, x)-negx);
	//CIO::message("pos_p_deriv=%e\n", featurevector[p-1]) ;
      } ;
      
      for (j=0; j<neg->get_M(); j++) {
	featurevector[p++]= - exp(neg->model_derivative_b(i, j, x)-negx);
	//CIO::message("pos_p_deriv=%e\n", featurevector[p-1]) ;
      } ;
    }
}

/*bool CHMM::save_top_features(CHMM* pos, CHMM* neg, FILE* dest)
{
	int totobs=pos->get_observations()->get_DIMENSION();
    int num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());
	
	CIO::message("saving %i features (size %i) for %i sequences.\n total file size should be: %i byte.\nwriting as doubles of size %i\n",num_features,sizeof(double)*num_features,totobs,sizeof(double)*num_features*totobs, sizeof(double));

	double* feature_vector=new double[num_features];

	for (int x=0; x<totobs; x++)
	{
		if (!(x % (totobs/10+1)))
			CIO::message("%02d%%.", (int) (100.0*x/totobs));
		else if (!(x % (totobs/200+1)))
			CIO::message(".");

		compute_top_feature_vector(pos, neg, x, feature_vector);
		fwrite(feature_vector, sizeof(double),num_features, dest);
	}

	CIO::message(".done.\n");
	delete[] feature_vector;

	return true;
}
*/

REAL* CTOPFeatures::set_feature_matrix()
{
	long len=0;

	num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	num_vectors=pos->get_observations()->get_DIMENSION();
	CIO::message("allocating top feature cache of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
	delete[] feature_matrix;
	feature_matrix=new REAL[num_features*num_vectors];

	CIO::message("calculating top feature matrix\n");

	for (long x=0; x<num_vectors; x++)
	{
		if (!(x % (num_vectors/10+1)))
			printf("%02d%%.", (int) (100.0*x/num_vectors));
		else if (!(x % (num_vectors/200+1)))
			printf(".");

		compute_feature_vector(&feature_matrix[x*num_features], x, len);
	}

	printf(".done.\n");
	
	num_vectors=get_number_of_examples() ;
	num_features=get_num_features() ;

	return feature_matrix;
}
