#include "lib/common.h"
#include "kernel/WeightedDegreeCharKernelPolyA.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreeCharKernelPolyA::CWeightedDegreeCharKernelPolyA(LONG size, double* w, INT d, INT max_mismatch_)
	: CCharKernel(size),weights(NULL),degree(d), max_mismatch(max_mismatch_), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), match_vector(NULL), 
	  lhs_sites(NULL), lhs_sites_num(NULL), rhs_sites(NULL), rhs_sites_num(NULL), lhs_num(0), rhs_num(0),
	  down_stream(20), up_stream(20)
{
	weights=new REAL[d*(1+max_mismatch)];
	assert(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];
}

CWeightedDegreeCharKernelPolyA::~CWeightedDegreeCharKernelPolyA() 
{
	cleanup();

	delete[] weights;
	weights=NULL;
}

void CWeightedDegreeCharKernelPolyA::remove_lhs() 
{ 
	if (lhs)
		cache_reset() ;
	
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
	delete[] match_vector ;
	
	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false ;
	sqrtdiag_lhs = NULL ;
	sqrtdiag_rhs = NULL ;
	match_vector = NULL ;
} ;

void CWeightedDegreeCharKernelPolyA::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	
	rhs = NULL ; 
	initialized = false ;
	sqrtdiag_rhs = NULL ;

}

char polyA_seqs[][7]={"aataaa", "attaaa", "agtaaa", "tataaa", "cataaa", "gataaa", 
					  "aatata", "aataca", "aataga", "aaaaag", "actaaa", "aagaaa",
					  "aatgaa", "tttaaa", "aaaaca"};

const INT num_polyA_seqs = 15 ;


INT* CWeightedDegreeCharKernelPolyA::find_site(char* seq, INT len, INT & num)
{
	INT buffer[1000] ;
	num = 0 ;
	
	for (INT i=down_stream; i<len-CMath::max(6,up_stream); i++)
		for (INT j=0; j<num_polyA_seqs; j++)
			if (strncmp(&seq[i], polyA_seqs[j],6)==0 && 
				((i>=99 && i<105) || (i>140 && i<160)))
			{
				buffer[num++]=i ;
				assert(num<1000) ;
			}
	
	INT * ret = new INT[num] ;
	for (INT i=0; i<num; i++)
		ret[i]=buffer[i] ;
	//CIO::message(M_DEBUG, "found %i sites\n", num) ;
	return ret ;
}

  
bool CWeightedDegreeCharKernelPolyA::init(CFeatures* l, CFeatures* r, bool do_init)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	CIO::message(M_DEBUG, "lhs_changed: %i\n", lhs_changed) ;
	CIO::message(M_DEBUG, "rhs_changed: %i\n", rhs_changed) ;
	
	if (lhs_changed) 
	{
		INT alen ;
		bool afree ;
		CHAR* avec=((CCharFeatures*) l)->get_feature_vector(0, alen, afree);
		delete[] match_vector ;
		match_vector=new bool[alen] ;
		((CCharFeatures*) l)->free_feature_vector(avec, 0, afree);		
	} 

	for (INT i=0; i<lhs_num; i++)
		delete[] lhs_sites[i] ;
	delete[] lhs_sites ;
	delete[] lhs_sites_num ;
	lhs_num = l->get_num_vectors() ;
	
	lhs_sites=new INT*[l->get_num_vectors()] ;
	lhs_sites_num=new INT[l->get_num_vectors()] ;
	for (INT i=0; i<l->get_num_vectors(); i++)
	{
		INT alen ;
		bool afree ;
		CHAR* avec=((CCharFeatures*) l)->get_feature_vector(i, alen, afree);
		lhs_sites[i]=find_site(avec, alen, lhs_sites_num[i]) ;
		((CCharFeatures*) l)->free_feature_vector(avec, i, afree);		
	}
	
	for (INT i=0; i<rhs_num; i++)
		delete[] rhs_sites[i] ;
	delete[] rhs_sites ; 
	delete[] rhs_sites_num ; 
	rhs_num = r->get_num_vectors() ;
	
	rhs_sites=new INT*[r->get_num_vectors()] ;
	rhs_sites_num=new INT[r->get_num_vectors()] ;
	for (INT i=0; i<r->get_num_vectors(); i++)
	{
		INT blen ;
		bool bfree ;
		CHAR* bvec=((CCharFeatures*) r)->get_feature_vector(i, blen, bfree);
		rhs_sites[i]=find_site(bvec, blen, rhs_sites_num[i]) ;
		((CCharFeatures*) r)->free_feature_vector(bvec, i, bfree);		
	}

	bool result=CCharKernel::init(l,r,do_init);
	initialized = false ;
	INT i;

	if (rhs_changed)
	{
		if (sqrtdiag_lhs != sqrtdiag_rhs)
			delete[] sqrtdiag_rhs;
		sqrtdiag_rhs=NULL ;
	}
	if (lhs_changed)
	{
		delete[] sqrtdiag_lhs;
		sqrtdiag_lhs=NULL ;
		sqrtdiag_lhs= new REAL[lhs->get_num_vectors()];
		assert(sqrtdiag_lhs) ;
		for (i=0; i<lhs->get_num_vectors(); i++)
			sqrtdiag_lhs[i]=1;
	}

	if (l==r)
		sqrtdiag_rhs=sqrtdiag_lhs;
	else if (rhs_changed)
	{
		sqrtdiag_rhs= new REAL[rhs->get_num_vectors()];
		assert(sqrtdiag_rhs) ;
		
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
	}

	assert(sqrtdiag_lhs);
	assert(sqrtdiag_rhs);

	if (lhs_changed)
	{
		this->lhs=(CCharFeatures*) l;
		this->rhs=(CCharFeatures*) l;
		
		//compute normalize to 1 values
		for (i=0; i<lhs->get_num_vectors(); i++)
		{
			sqrtdiag_lhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_lhs[i]==0)
				sqrtdiag_lhs[i]=1e-16;
		}
	};
	
	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if ((sqrtdiag_lhs!=sqrtdiag_rhs) & rhs_changed)
	{
		this->lhs=(CCharFeatures*) r;
		this->rhs=(CCharFeatures*) r;

		INT **lhs_sites_old=lhs_sites ;
		INT *lhs_sites_num_old=lhs_sites_num ;
		lhs_sites = rhs_sites ;
		lhs_sites_num = rhs_sites_num ;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
			sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
		lhs_sites = lhs_sites_old ;
		lhs_sites_num = lhs_sites_num_old ;
	}
	
	this->lhs=(CCharFeatures*) l;
	this->rhs=(CCharFeatures*) r;

	initialized = true ;
	return result;
}

void CWeightedDegreeCharKernelPolyA::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	delete[] match_vector ;
	match_vector = NULL;

	for (INT i=0; i<lhs_num; i++)
	{
		delete[] lhs_sites[i] ;
	}
	for (INT i=0; i<rhs_num; i++)
	{
		delete[] rhs_sites[i] ;
	}

	delete[] lhs_sites ;
	lhs_sites = NULL;

	delete[] lhs_sites_num ;
	lhs_sites_num = NULL;

	delete[] rhs_sites ;
	rhs_sites = NULL;

	delete[] rhs_sites_num ;
	rhs_sites_num = NULL;
}

bool CWeightedDegreeCharKernelPolyA::load_init(FILE* src)
{
	return false;
}

bool CWeightedDegreeCharKernelPolyA::save_init(FILE* dest)
{
	return false;
}

REAL CWeightedDegreeCharKernelPolyA::compute(INT idx_a, INT idx_b)
{
	REAL sum = 0 ;
    INT num = 1 ;//rhs_sites_num[idx_b]*lhs_sites_num[idx_a] ;
	
	//fprintf(stderr,"idxa=%i idxb=%i\n", idx_a, idx_b) ;
	for (INT i=0; i<lhs_sites_num[idx_a]; i++)
		for (INT j=0; j<rhs_sites_num[idx_b]; j++)
		{
			sum+=compute_with_offset(idx_a, lhs_sites[idx_a][i], idx_b, rhs_sites[idx_b][j]) ;
			//fprintf(stderr,"sum=%f l1=%i l2=%i\n", sum, lhs_sites[idx_a][i], rhs_sites[idx_b][j]) ;
		}
	
	
	REAL sqrt_a= 1 ;
	REAL sqrt_b= 1 ;
	if (initialized)
    {
		sqrt_a=sqrtdiag_lhs[idx_a] ;
		sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;
	
	REAL sqrt_both=sqrt_a*sqrt_b;

	return sum/(sqrt_both*num) ;
}

REAL CWeightedDegreeCharKernelPolyA::compute_with_offset(INT idx_a, INT offset_a, INT idx_b, INT offset_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

  double sum=0;

  for (INT i=-down_stream; i<up_stream; i++)
	  match_vector[i+down_stream]=(avec[i+offset_a]!=bvec[i+offset_b]) ;
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  for (INT i=0; i<up_stream+down_stream-degree; i++)
  {
	  INT mismatches=0;
	  
	  for (INT j=0; j<degree; j++)
	  {
		  if (match_vector[i+j])
		  {
			  mismatches++ ;
			  if (mismatches>max_mismatch)
				  break ;
		  } ;
		  sum += weights[j+degree*mismatches];
	  }
  }

  return sum ;
}
