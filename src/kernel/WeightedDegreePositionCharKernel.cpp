#include "lib/common.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreePositionCharKernel::CWeightedDegreePositionCharKernel(LONG size, REAL* w, INT d, 
																	 INT max_mismatch_, INT * shift_, 
																	 INT shift_len_, bool use_norm,
																	 INT mkl_stepsize_)
	: CCharKernel(size),weights(NULL),position_weights(NULL),counts(NULL),
	  weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), 
	  max_mismatch(max_mismatch_), seq_length(0), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	  match_vector(NULL), use_normalization(use_norm)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION;
	lhs=NULL;
	rhs=NULL;

	weights=new REAL[d*(1+max_mismatch)];
	counts = new INT[d*(1+max_mismatch)];

	assert(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

	shift_len = shift_len_ ;
	shift = new INT[shift_len] ;
	max_shift = 0 ;
	
	for (INT i=0; i<shift_len; i++)
	{
		shift[i] = shift_[i] ;
		if (shift[i]>max_shift)
			max_shift = shift[i] ;
	} ;
	assert(max_shift>=0 && max_shift<=shift_len) ;

	length=0 ;
	trees=NULL ;

	tree_initialized=false ;
}

CWeightedDegreePositionCharKernel::~CWeightedDegreePositionCharKernel() 
{
	delete[] shift;
	shift = NULL;

	delete[] counts;
	counts = NULL;

	delete[] weights ;
	weights=NULL ;

	delete[] position_weights ;
	position_weights=NULL ;

	delete[] weights_buffer ;
	weights_buffer = NULL ;
	
	cleanup();
}

void CWeightedDegreePositionCharKernel::remove_lhs() 
{ 
	delete_optimization();

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
	
	if (trees!=NULL)
	{
		for (INT i=0; i<seq_length; i++)
		{
			delete trees[i];
			trees[i]=NULL;
		}
		delete[] trees ;
		trees=NULL ;
	} ;

} ;

void CWeightedDegreePositionCharKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

  
bool CWeightedDegreePositionCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
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
		
		if (trees)
		{
			delete_tree() ;
			for (INT i=0; i<seq_length; i++)
			{
				delete trees[i] ;
				trees[i]=NULL ;
			}
			delete[] trees ;
			trees=NULL ;
		}
		
#ifdef OSF1
		trees=new (struct SuffixTree**)[alen] ;		
#else
		trees=new (struct SuffixTree*)[alen] ;		
#endif
		for (INT i=0; i<alen; i++)
		{
			trees[i]=new struct SuffixTree ;
			trees[i]->weight=0 ;
			trees[i]->has_floats=false ;
			trees[i]->usage=0;
			for (INT j=0; j<4; j++)
				trees[i]->childs[j]=NULL ;
		} 
		seq_length = alen ;
		((CCharFeatures*) l)->free_feature_vector(avec, 0, afree);
	} 

	bool result=CCharKernel::init(l,r,do_init);
	initialized = false ;
	INT i;

	if (use_normalization)
	{
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
		}

		// if lhs is different from rhs (train/test data)
		// compute also the normalization for rhs
		if ((sqrtdiag_lhs!=sqrtdiag_rhs) & rhs_changed)
		{
			this->lhs=(CCharFeatures*) r;
			this->rhs=(CCharFeatures*) r;

			//compute normalize to 1 values
			for (i=0; i<rhs->get_num_vectors(); i++)
			{
				sqrtdiag_rhs[i]=sqrt(compute(i,i));

				//trap divide by zero exception
				if (sqrtdiag_rhs[i]==0)
					sqrtdiag_rhs[i]=1e-16;
			}
		}
	}
	
	this->lhs=(CCharFeatures*) l;
	this->rhs=(CCharFeatures*) r;

	initialized = true ;
	return result;
}

void CWeightedDegreePositionCharKernel::cleanup()
{
	delete_optimization();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	if (trees!=NULL)
	{
		for (INT i=0; i<seq_length; i++)
		{
			delete trees[i];
			trees[i]=NULL;
		}
		delete[] trees;
		trees=NULL;
	}

	lhs = NULL;
	rhs = NULL;

	seq_length = 0;
	initialized = false;
	match_vector = NULL;
	tree_initialized = false;
}

bool CWeightedDegreePositionCharKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    UINT intlen=0;
    UINT endian=0;
    UINT fourcc=0;
    UINT doublelen=0;
    INT d=1;

    assert(fread(&intlen, sizeof(BYTE), 1, src)==1);
    assert(fread(&doublelen, sizeof(BYTE), 1, src)==1);
    assert(fread(&endian, (UINT) intlen, 1, src)== 1);
    assert(fread(&fourcc, (UINT) intlen, 1, src)==1);
    assert(fread(&d, (UINT) intlen, 1, src)==1);
	double* w= new double[d];
	assert(w) ;
	
    assert(fread(w, sizeof(double), d, src)==(UINT) d) ;

	for (INT i=0; i<d; i++)
		weights[i]=w[i];

    CIO::message(M_INFO, "detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	
	return true;
}

bool CWeightedDegreePositionCharKernel::save_init(FILE* /*dest*/)
{
	return false;
}

bool CWeightedDegreePositionCharKernel::init_optimization(INT count, INT * IDX, REAL * alphas)
{
	if (max_mismatch!=0)
	{
		CIO::message(M_ERROR, "CWeightedDegreePositionCharKernel optimization not implemented for mismatch!=0\n") ;
		return false ;
	}

	delete_optimization();
	
	CIO::message(M_DEBUG, "initializing CWeightedDegreePositionCharKernel optimization\n") ;
	int i=0;
	for (i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::progress(i,0,count);
		add_example_to_tree(IDX[i], alphas[i]) ;
	}
	CIO::message(M_MESSAGEONLY, "done.           \n");
	
	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreePositionCharKernel::delete_optimization() 
{ 
	CIO::message(M_DEBUG, "deleting CWeightedDegreePositionCharKernel optimization\n");

	if (get_is_initialized())
	{
		delete_tree(NULL); 
		set_is_initialized(false);
		return true;
	}
	
	return false;
}


/* \hat K_l(x,y) = sum_{i=0}^l (k_i(x,y)+k_i(y,x)) / (2*(i+1))

   k_i(x,y) = k(x(1:end-i),y(1+i:end)) where k is the standard 
                                     weighted degree kernel
   K_0(x,y) = k(x,y)
   K_1(x,y) = k(x,y) + (k(x(1:end-1),y(2:end)) + k(x(2:end),y(1:end-1)))/2
   ...
*/
  
REAL CWeightedDegreePositionCharKernel::compute2(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen == blen);
  assert(shift_len == alen) ;

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized && use_normalization)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  REAL sum0=0 ;
  REAL* sum1=new REAL[max_shift] ;
  REAL* sum2=new REAL[max_shift] ;
  for (INT i=0; i<max_shift; i++)
  {
	  sum1[i]=0 ;
	  sum2[i]=0 ;
  }
  
  // no shift
  for (INT i=0; i<alen; i++)
  {
	  INT mismatches=0;
	  for (INT j=0; (j<degree) && (i+j<alen) && mismatches<=max_mismatch; j++)
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
  for (INT i=0; i<alen; i++)
  {
	  for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
	  {
		  INT mismatches=0;
		  for (INT j=0; (j<degree) && (i+j+k<alen) && mismatches<=max_mismatch; j++)
		  {
			  if (avec[i+j+k]!=bvec[i+j])
			  {
				  mismatches++ ;
				  if (mismatches>max_mismatch)
					  break ;
			  } ;
			  sum1[k-1] += weights[j+degree*mismatches];
		  }
	  } ;
  }

  // shift in sequence b
  for (INT i=0; i<alen; i++)
  {
	  for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
	  {
		  INT mismatches=0;
		  for (INT j=0; (j<degree) && (i+j+k<alen) && mismatches<=max_mismatch; j++)
		  {
			  if (avec[i+j]!=bvec[i+j+k])
			  {
				  mismatches++ ;
				  if (mismatches>max_mismatch)
					  break ;
			  } ;
			  sum2[k-1] += weights[j+degree*mismatches];
		  }
	  } ;
  }
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  REAL result = sum0 ;
  for (INT i=0; i<max_shift; i++)
	  result += (sum1[i]+sum2[i])/(2*(i+1)) ;

  delete[] sum1 ;
  delete[] sum2 ;
  return (double) result/sqrt_both;
}

REAL CWeightedDegreePositionCharKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	REAL sum0=0 ;
	REAL *sum1=new REAL[max_shift] ;
	for (INT i=0; i<max_shift; i++)
		sum1[i]=0 ;
	
	// no shift
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		INT mismatches=0;
		REAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
			{
				mismatches++ ;
				if (mismatches>max_mismatch)
					break ;
			} ;
			sumi += weights[j+degree*mismatches];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;
	
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			REAL sumi = 0.0 ;
			// shift in sequence a
			INT mismatches=0;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi += weights[j+degree*mismatches];
			}
			// shift in sequence b
			mismatches=0;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi += weights[j+degree*mismatches];
			}
			if (position_weights!=NULL)
				sum1[k-1] += position_weights[i]*sumi ;
			else
				sum1[k-1] += sumi ;
		} ;
	}

	REAL result = sum0 ;
	for (INT i=0; i<max_shift; i++)
		result += sum1[i]/(2*(i+1)) ;

	delete[] sum1 ;
	return result ;
}

REAL CWeightedDegreePositionCharKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	REAL sum0=0 ;
	REAL *sum1=new REAL[max_shift] ;
	for (INT i=0; i<max_shift; i++)
		sum1[i]=0 ;
	
	// no shift
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		REAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;
	
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			REAL sumi = 0.0 ;
			// shift in sequence a
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi += weights[j];
			}
			// shift in sequence b
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi += weights[j];
			}
			if (position_weights!=NULL)
				sum1[k-1] += position_weights[i]*sumi ;
			else
				sum1[k-1] += sumi ;
		} ;
	}

	REAL result = sum0 ;
	for (INT i=0; i<max_shift; i++)
		result += sum1[i]/(2*(i+1)) ;

	delete[] sum1 ;
	return result ;
}

REAL CWeightedDegreePositionCharKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	REAL sum0=0 ;
	REAL *sum1=new REAL[max_shift] ;
	for (INT i=0; i<max_shift; i++)
		sum1[i]=0 ;
	
	// no shift
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		REAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[i*degree+j];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;
	
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			REAL sumi = 0.0 ;
			// shift in sequence a
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi += weights[i*degree+j];
			}
			// shift in sequence b
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi += weights[i*degree+j];
			}
			if (position_weights!=NULL)
				sum1[k-1] += position_weights[i]*sumi ;
			else
				sum1[k-1] += sumi ;
		} ;
	}

	REAL result = sum0 ;
	for (INT i=0; i<max_shift; i++)
		result += sum1[i]/(2*(i+1)) ;

	delete[] sum1 ;
	return result ;
}


REAL CWeightedDegreePositionCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen == blen);
  assert(shift_len == alen) ;

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized && use_normalization)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;
  REAL sqrt_both=sqrt_a*sqrt_b;

  REAL result = 0 ;
  if (max_mismatch > 0)
	  result = compute_with_mismatch(avec, alen, bvec, blen) ;
  else if (length==0)
	  result = compute_without_mismatch(avec, alen, bvec, blen) ;
  else
	  result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
  result/=sqrt_both;
  
  //REAL result2 = compute2(idx_a,idx_b) ;
  //assert(fabs(result-result2)<1e-6);
  
  return result ;
  
}

void CWeightedDegreePositionCharKernel::add_example_to_tree(INT idx, REAL alpha) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
	for (INT i=0; i<len; i++)
	{
		if (char_vec[i]=='A') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='C') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='G') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='T') { vec[i]=3 ; continue ; } ;
		if (char_vec[i]=='a') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='c') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='g') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='t') { vec[i]=3 ; continue ; } ;
		vec[i]=0 ;
	} ;
		
	for (INT i=0; i<len; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		
		for (INT j=0; (j<degree) && (i+j<len); j++)
		{
			if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
			{
				tree=tree->childs[vec[i+j]] ;
				tree->weight += alpha ;
			} else 
			{
				if ((j==degree-1) && (tree->has_floats))
				{
					tree->child_weights[vec[i+j]] += alpha ;
					break ;
				}
				else
				{
					if (j==degree-1)
					{
						assert(!tree->has_floats) ;
						tree->has_floats=true ;
						for (INT k=0; k<4; k++)
						{
							assert(tree->childs[k]==NULL) ;
							tree->child_weights[k] =0 ;
						}
						tree->child_weights[vec[i+j]] += alpha ;
						break ;
					}
					else
					{
						assert(!tree->has_floats) ;
						tree->childs[vec[i+j]]=new struct SuffixTree ;
						assert(tree->childs[vec[i+j]]!=NULL) ;
						tree=tree->childs[vec[i+j]] ;
						for (INT k=0; k<4; k++)
							tree->childs[k]=NULL ;
						tree->weight = alpha ;
						tree->has_floats=false ;
						tree->usage=0 ;
					} ;
				} ;
			} ;
		} ;
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

REAL CWeightedDegreePositionCharKernel::compute_by_tree(INT idx) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
	{
		if (char_vec[i]=='A') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='C') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='G') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='T') { vec[i]=3 ; continue ; } ;
		if (char_vec[i]=='a') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='c') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='g') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='t') { vec[i]=3 ; continue ; } ;
		vec[i]=0 ;
	} ;
	
	REAL sum = 0 ;
	for (INT i=0; i<len; i++)
		sum += compute_by_tree_helper(vec, len, i, i, i) ;

	for (INT i=0; i<len; i++)
		for (INT k=1; (k<=shift[i]) && (i+k<len); k++)
		{
			sum+=compute_by_tree_helper(vec, len, i, i+k, i)/(2*k) ;
			sum+=compute_by_tree_helper(vec, len, i+k, i, i)/(2*k) ;
		}
	
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;

	if (use_normalization)
		return sum/sqrtdiag_rhs[idx] ;
	else
		return sum ;
}

void CWeightedDegreePositionCharKernel::compute_by_tree(INT idx, REAL* LevelContrib) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
	{
		if (char_vec[i]=='A') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='C') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='G') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='T') { vec[i]=3 ; continue ; } ;
		if (char_vec[i]=='a') { vec[i]=0 ; continue ; } ;
		if (char_vec[i]=='c') { vec[i]=1 ; continue ; } ;
		if (char_vec[i]=='g') { vec[i]=2 ; continue ; } ;
		if (char_vec[i]=='t') { vec[i]=3 ; continue ; } ;
		vec[i]=0 ;
	} ;

	REAL factor = 1.0 ;

	if (use_normalization)
		factor = 1.0/sqrtdiag_rhs[idx] ;

	for (INT i=0; i<len; i++)
		compute_by_tree_helper(vec, len, i, i, i, LevelContrib, factor) ;
	
	for (INT i=0; i<len; i++)
		for (INT k=1; (k<=shift[i]) && (i+k<len); k++)
		{
			compute_by_tree_helper(vec, len, i, i+k, i, LevelContrib, factor/(2*k)) ;
			compute_by_tree_helper(vec, len, i+k, i, i, LevelContrib, factor/(2*k)) ;
		}
	
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
}

REAL CWeightedDegreePositionCharKernel::compute_abs_weights_tree(struct SuffixTree* p_tree) 
{
	REAL ret=0 ;

	if (p_tree==NULL)
		return 0 ;
		
	if (p_tree->has_floats)
	{
		for (INT k=0; k<4; k++)
			ret+=(p_tree->child_weights[k]) ;
		
		return ret ;
	}

	ret+=(p_tree->weight) ;

	for (INT i=0; i<4; i++)
		if (p_tree->childs[i]!=NULL)
			ret += compute_abs_weights_tree(p_tree->childs[i])  ;

	return ret ;
}

REAL *CWeightedDegreePositionCharKernel::compute_abs_weights(int &len) 
{
	REAL * sum=new REAL[seq_length*4] ;
	for (INT i=0; i<seq_length*4; i++)
		sum[i]=0 ;
	len=seq_length ;
	
	for (INT i=0; i<seq_length; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		for (INT k=0; k<4; k++)
			sum[i*4+k]=compute_abs_weights_tree(tree->childs[k]) ;
	}

	return sum ;
}

void CWeightedDegreePositionCharKernel::delete_tree(struct SuffixTree * p_tree)
{
	if (p_tree==NULL)
	{
		if (trees==NULL)
			return;

		for (INT i=0; i<seq_length; i++)
		{
			delete_tree(trees[i]);

			trees[i]->has_floats=false;
			trees[i]->usage=0;

			for (INT k=0; k<4; k++)
				trees[i]->childs[k]=NULL;
		}

		tree_initialized=false;
		return;
	}

	if (p_tree->has_floats)
		return;
	
	for (INT i=0; i<4; i++)
	{
		if (p_tree->childs[i]!=NULL)
		{
			delete_tree(p_tree->childs[i]);
			delete p_tree->childs[i];
			p_tree->childs[i]=NULL;
		} 
		p_tree->weight=0;
	}
} 

bool CWeightedDegreePositionCharKernel::set_weights(REAL* ws, INT d, INT len)
{
	CIO::message(M_DEBUG, "degree = %i  d=%i\n", degree, d) ;
	degree = d ;
	length=len;
	
	if (len <= 0)
		len=1;
	
	delete[] weights;
	weights=new REAL[d*len];
	
	if (weights)
	{
		for (int i=0; i<degree*len; i++)
			weights[i]=ws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreePositionCharKernel::set_position_weights(REAL* pws, INT len)
{
	if (len==0)
	{
		delete[] position_weights ;
		position_weights = NULL ;
	}
	if (seq_length==0)
		seq_length = len ;

    if (seq_length!=len) 
	{
		CIO::message(M_ERROR, "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
	}
	delete[] position_weights;
	position_weights=new REAL[len];
	
	if (position_weights)
	{
		for (int i=0; i<len; i++)
			position_weights[i]=pws[i];
		return true;
	}
	else
		return false;
}
