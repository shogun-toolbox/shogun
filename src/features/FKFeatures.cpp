#include "FKFeatures.h"
#include "lib/io.h"
#include <assert.h>

CFKFeatures::CFKFeatures(CHMM* p, CHMM* n, REAL a)
{
    assert(p!=NULL && n!=NULL);
    pos=p;
    neg=n;
	weight_a=a;
    set_num_vectors(0);
    if (pos && pos->get_observations())
	set_num_vectors(pos->get_observations()->get_DIMENSION());

    CIO::message("pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i]\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M()) ;

    if (pos && neg)
	num_features=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ;
}

 CFKFeatures::CFKFeatures(const CFKFeatures &orig): 
	CRealFeatures(orig), pos(orig.pos), neg(orig.neg), weight_a(orig.weight_a)
{ 
}

CFKFeatures::~CFKFeatures()
{
}

void CFKFeatures::set_models(CHMM* p, CHMM* n, REAL a)
{
  pos=p ; 
  neg=n ;
  weight_a=a;

  delete[] feature_matrix  ;
  feature_matrix=NULL ;
  CIO::message("pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i]\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M()) ;
  if (pos && neg)
    num_features=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ;
}

int CFKFeatures::get_label(long idx)
{
    
  if (pos && pos->get_observations())
    return pos->get_observations()->get_label(idx) ;

  assert(0) ;
  return 0 ;
}

CFeatures* CFKFeatures::duplicate() const
{
	return new CFKFeatures(*this);
}

REAL* CFKFeatures::compute_feature_vector(long num, long &len)
{
  REAL* featurevector;
  
  //CIO::message("allocating %.2f M for FK feature vector cache\n", 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()));
  
  featurevector=new REAL[ 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ];
  
  if (!featurevector)
    return NULL;
  
  compute_feature_vector(featurevector, num, len);

  return featurevector;
}

void CFKFeatures::compute_feature_vector(REAL* featurevector, long num, long& len)
{
	long i,j,p=0,x=num;

	double posx=pos->model_probability(x);
	double negx=neg->model_probability(x);

	len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());
	//  CIO::message("len=%i\n",len) ;

	featurevector[p++]=(exp(posx)-exp(negx))/(weight_a*exp(posx)+(1-weight_a)*exp(negx));
//	CIO::message("posx+negx=%f\n", featurevector[0]);

	//first do positive model
	for (i=0; i<pos->get_N(); i++)
	{
		featurevector[p++]=weight_a*exp(pos->model_derivative_p(i, x)-posx);
//		CIO::message("pos_p_deriv=%e\n", featurevector[p-1]) ;
		featurevector[p++]=weight_a*exp(pos->model_derivative_q(i, x)-posx);
//		CIO::message("pos_q_deriv=%e\n", featurevector[p-1]) ;

		for (j=0; j<pos->get_N(); j++) {
			featurevector[p++]=weight_a*exp(pos->model_derivative_a(i, j, x)-posx);
//			CIO::message("pos_a_deriv[%i]=%e\n", j, featurevector[p-1]) ;
		}

		for (j=0; j<pos->get_M(); j++) {
			featurevector[p++]=weight_a*exp(pos->model_derivative_b(i, j, x)-posx);
//			CIO::message("pos_b_deriv[%i]=%e\n", j, featurevector[p-1]) ;
		} 

	}

	//then do negative
	for (i=0; i<neg->get_N(); i++)
	{
		featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_p(i, x)-negx);
//		CIO::message("neg_p_deriv=%e\n", featurevector[p-1]) ;
		featurevector[p++]= (1-weight_a)* exp(neg->model_derivative_q(i, x)-negx);
//		CIO::message("neg_q_deriv=%e\n", featurevector[p-1]) ;

		for (j=0; j<neg->get_N(); j++) {
			featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_a(i, j, x)-negx);
//			CIO::message("neg_a_deriv=%e\n", featurevector[p-1]) ;
		}

		for (j=0; j<neg->get_M(); j++) {
			featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_b(i, j, x)-negx);
//			CIO::message("neg_b_deriv=%e\n", featurevector[p-1]) ;
		}
	}
}

REAL* CFKFeatures::set_feature_matrix()
{
	long len=0;

	num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	num_vectors=pos->get_observations()->get_DIMENSION();
	CIO::message("allocating FK feature cache of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
	delete[] feature_matrix;
	feature_matrix=new REAL[num_features*num_vectors];

	CIO::message("calculating FK feature matrix\n");

	for (long x=0; x<num_vectors; x++)
	{
		if (!(x % (num_vectors/10+1)))
			printf("%02d%%.", (int) (100.0*x/num_vectors));
		else if (!(x % (num_vectors/200+1)))
			printf(".");

		compute_feature_vector(&feature_matrix[x*num_features], x, len);
	}

	printf(".done.\n");
	
	num_vectors=get_num_vectors() ;
	num_features=get_num_features() ;

	return feature_matrix;
}
