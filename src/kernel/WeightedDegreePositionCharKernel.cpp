#include "lib/common.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreePositionCharKernel::CWeightedDegreePositionCharKernel(LONG size, double* w, INT d, INT max_mismatch_, INT * shift_, INT shift_len_)
	: CCharKernel(size),weights(NULL),degree(d), max_mismatch(max_mismatch_), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
	lhs=NULL ;
	rhs=NULL ;

	weights=new REAL[d*(1+max_mismatch)];
	assert(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

	shift_len = shift_len ;
	shift = new INT[shift_len] ;
	max_shift = 0 ;
	
	for (INT i=0; i<shift_len; i++)
	{
		shift[i] = shift_[i] ;
		if (shift[i]>max_shift)
			max_shift = shift[i] ;
	} ;
	assert(max_shift>0 && max_shift<=shift_len) ;
}

CWeightedDegreePositionCharKernel::~CWeightedDegreePositionCharKernel() 
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
	delete[] shift ;

	cleanup();
}
  
bool CWeightedDegreePositionCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	CIO::message("lhs_changed: %i\n", lhs_changed) ;
	CIO::message("rhs_changed: %i\n", rhs_changed) ;

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
			sqrtdiag_lhs[i]=sqrt(compute(i,i));
	};
	
	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if ((sqrtdiag_lhs!=sqrtdiag_rhs) & rhs_changed)
	{
		this->lhs=(CCharFeatures*) r;
		this->rhs=(CCharFeatures*) r;
		
		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=sqrt(compute(i,i));
	}
	
	this->lhs=(CCharFeatures*) l;
	this->rhs=(CCharFeatures*) r;

	initialized = true ;
	return result;
}
void CWeightedDegreePositionCharKernel::cleanup()
{
	delete[] weights;
	weights=NULL;
}

bool CWeightedDegreePositionCharKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    UINT intlen=0;
    UINT endian=0;
    UINT fourcc=0;
    UINT doublelen=0;
	double* w=NULL;
    INT d=1;

    assert(fread(&intlen, sizeof(BYTE), 1, src)==1);
    assert(fread(&doublelen, sizeof(BYTE), 1, src)==1);
    assert(fread(&endian, (UINT) intlen, 1, src)== 1);
    assert(fread(&fourcc, (UINT) intlen, 1, src)==1);
    assert(fread(&d, (UINT) intlen, 1, src)==1);
	double* weights= new double[d];
	assert(weights) ;
	
    assert(fread(w, sizeof(double), d, src)==(UINT) d) ;

	for (INT i=0; i<d; i++)
		weights[i]=w[i];

    CIO::message("detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	return true;
}

bool CWeightedDegreePositionCharKernel::save_init(FILE* dest)
{
	return false;
}

/* \hat K_l(x,y) = sum_{i=0}^l (k_i(x,y)+k_i(y,x)) / (2*(i+1))

   k_i(x,y) = k(x(1:end-i),y(1+i:end)) where k is the standard 
                                     weighted degree kernel
   K_0(x,y) = k(x,y)
   K_1(x,y) = k(x,y) + (k(x(1:end-1),y(2:end)) + k(x(2:end),y(1:end-1)))/2
   ...
*/
  
REAL CWeightedDegreePositionCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen == blen);
  assert(shift_len == alen - degree) ;

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  double sum0=0 ;
  double sum1[max_shift] ;
  double sum2[max_shift] ;
  for (INT i=0; i<max_shift; i++)
  {
	  sum1[i]=0 ;
	  sum2[i]=0 ;
  }
  
  // no shift
  for (INT i=0; i<alen-degree; i++)
  {
	  INT mismatches=0;
	  for (INT j=0; j<degree && mismatches<=max_mismatch; j++)
	  {
		  if (avec[i+j]!=bvec[i+j])
		  {
			  mismatches++ ;
			  if (mismatches>max_mismatch)
				  break ;
		  } ;
		  sum0 += weights[j+degree*mismatches];
	  }
  } ;
  
  // shift in sequence a
  for (INT i=0; i<alen-degree; i++)
  {
	  for (INT k=1; (k<=shift[i]) && (i<alen-degree-k); k++)
	  {
		  INT mismatches=0;
		  INT pos = k + max_shift ;
		  for (INT j=0; j<degree && mismatches<=max_mismatch; j++)
		  {
			  if (avec[i+j+k]!=bvec[i+j])
			  {
				  mismatches++ ;
				  if (mismatches>max_mismatch)
					  break ;
			  } ;
			  sum1[pos] += weights[j+degree*mismatches];
		  }
	  } ;
  }

  // shift in sequence b
  for (INT i=0; i<alen-degree; i++)
  {
	  for (INT k=1; (k<=shift[i]) && (i<alen-degree-k); k++)
	  {
		  INT mismatches=0;
		  INT pos = k + max_shift ;
		  for (INT j=0; j<degree && mismatches<=max_mismatch; j++)
		  {
			  if (avec[i+j]!=bvec[i+j+k])
			  {
				  mismatches++ ;
				  if (mismatches>max_mismatch)
					  break ;
			  } ;
			  sum2[pos] += weights[j+degree*mismatches];
		  }
	  } ;
  }
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  REAL result = sum0 ;
  for (INT i=0; i<max_shift; i++)
	  result += (sum1[i]+sum2[i])/(2*(i+1)) ;
  
  return (double) result/sqrt_both;
}
