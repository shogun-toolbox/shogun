#include "lib/common.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(LONG size, double* w, INT d, INT max_mismatch_, bool use_norm)
	: CCharKernel(size),weights(NULL),degree(d), max_mismatch(max_mismatch_), seq_length(0),
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), match_vector(NULL), use_normalization(use_norm)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION;
	lhs=NULL;
	rhs=NULL;

	weights=new REAL[d*(1+max_mismatch)];
	assert(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

	length = 0;
	trees=NULL;
	tree_initialized=false ;
}

CWeightedDegreeCharKernel::~CWeightedDegreeCharKernel() 
{
	cleanup();

	delete[] weights;
	weights=NULL;
}

void CWeightedDegreeCharKernel::remove_lhs() 
{ 
	delete_optimization();

	if (lhs)
		cache_reset();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
	delete[] match_vector;

	lhs = NULL; 
	rhs = NULL; 
	initialized = false;
	sqrtdiag_lhs = NULL;
	sqrtdiag_rhs = NULL;
	match_vector = NULL;
	
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
}

void CWeightedDegreeCharKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

  
bool CWeightedDegreeCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
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

void CWeightedDegreeCharKernel::cleanup()
{
	delete_optimization();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	delete[] match_vector;
	match_vector = NULL;

	initialized=false;

	if (trees!=NULL)
	{
		for (INT i=0; i<seq_length; i++)
		{
			delete trees[i];
			trees[i]=NULL;
		}
		delete[] trees ;
		trees=NULL;
	}
	seq_length=0;
}

bool CWeightedDegreeCharKernel::load_init(FILE* src)
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

bool CWeightedDegreeCharKernel::save_init(FILE* dest)
{
	return false;
}
  

bool CWeightedDegreeCharKernel::init_optimization(INT count, INT * IDX, REAL * weights)
{
	if (max_mismatch!=0)
	{
		CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not implemented for mismatch!=0\n") ;
		return false ;
	}

	delete_optimization();
	
	CIO::message(M_DEBUG, "initializing CWeightedDegreeCharKernel optimization\n") ;
	int i=0;
	for (i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::message(M_PROGRESS, "%3i%%  \r", 100*i/(count+1)) ;
		add_example_to_tree(IDX[i], weights[i]) ;
	}
	CIO::message(M_PROGRESS, "done.           \n");
	
	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreeCharKernel::delete_optimization() 
{ 
	CIO::message(M_DEBUG, "deleting CWeightedDegreeCharKernel optimization\n");

	if (get_is_initialized())
	{
		delete_tree(NULL); 
		set_is_initialized(false);
		return true;
	}
	
	return false;
}

REAL CWeightedDegreeCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

	//CIO::message(M_DEBUG, "COMPUTE(%d,%d)\n",idx_a, idx_b);
  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized && use_normalization)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  double sum=0;

  for (INT i=0; i<alen; i++)
	  match_vector[i]=(avec[i]!=bvec[i]) ;
  
  if (length==0 || max_mismatch > 0)
  {
	  for (INT i=0; i<alen-degree; i++)
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
  }
  else
  {
	  for (INT i=0; i<alen-degree; i++)
	  {
		  for (INT j=0; j<degree; j++)
		  {
			  if (!match_vector[i+j])
				  break;
			  sum += weights[i*degree+j];
		  }
	  }
  }
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) sum/sqrt_both;
}

void CWeightedDegreeCharKernel::add_example_to_tree(INT idx, REAL weight) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	if (use_normalization)
		weight /=  sqrtdiag_lhs[idx] ;
	
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
		
	if (length == 0 || max_mismatch > 0)
	{
		for (INT i=0; i<len-degree; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			for (INT j=0; j<degree; j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					tree->weight += weight*weights[j];
				} else 
					if ((j==degree-1) && (tree->has_floats))
					{
						tree->child_weights[vec[i+j]] += weight*weights[j];
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
							tree->child_weights[vec[i+j]] += weight*weights[j];
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
							tree->weight = weight*weights[j] ;
							tree->has_floats=false ;
							tree->usage=0 ;
						} ;
					} ;
			} ;
		}
	}
	else
	{
		for (INT i=0; i<len-degree; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			for (INT j=0; j<degree; j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					tree->weight += weight*weights[i*degree + j];
				} else 
					if ((j==degree-1) && (tree->has_floats))
					{
						tree->child_weights[vec[i+j]] += weight*weights[i*degree + j];
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
							tree->child_weights[vec[i+j]] += weight*weights[i*degree + j];
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
							tree->weight = weight*weights[i*degree + j] ;
							tree->has_floats=false ;
							tree->usage=0 ;
						} ;
					} ;
			} ;
		}
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

REAL CWeightedDegreeCharKernel::compute_by_tree(INT idx) 
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
		
	REAL sum=0 ;
	for (INT i=0; i<len-degree; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		
		for (INT j=0; j<degree; j++)
		{
			if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
			{
				tree=tree->childs[vec[i+j]] ;
				sum += tree->weight ;
			} else
				if (tree->has_floats)
				{
					sum += tree->child_weights[vec[i+j]] ;
					break ;
				} else
					break ;
		} 
	}
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;

	if (use_normalization)
		 sum /= sqrtdiag_rhs[idx] ;

	return sum;
}

void CWeightedDegreeCharKernel::compute_by_tree(INT idx, REAL* LevelContrib) 
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
		
	for (INT j=0; j<degree; j++)
		LevelContrib[j]=0 ;
	
	for (INT i=0; i<len-degree; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		
		for (INT j=0; j<degree; j++)
		{
			if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
			{
				tree=tree->childs[vec[i+j]] ;
				LevelContrib[j] += tree->weight ;
			} else
				if (tree->has_floats)
				{
					LevelContrib[j] += tree->child_weights[vec[i+j]] ;
					break ;
				} else
					break ;
		} 
	}
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;

	if (use_normalization)
	{
		for (INT j=0; j<degree; j++)
			LevelContrib[j] /= sqrtdiag_rhs[idx] ;
	}
}

REAL CWeightedDegreeCharKernel::compute_abs_weights_tree(struct SuffixTree* p_tree) 
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

REAL *CWeightedDegreeCharKernel::compute_abs_weights(int &len) 
{
	REAL * sum=new REAL[seq_length*4] ;
	for (INT i=0; i<seq_length*4; i++)
		sum[i]=0 ;
	len=seq_length ;
	
	for (INT i=0; i<seq_length-degree; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		for (INT k=0; k<4; k++)
			sum[i*4+k]=compute_abs_weights_tree(tree->childs[k]) ;
	}

	return sum ;
}

void CWeightedDegreeCharKernel::count_tree_usage(INT idx) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
	{
		vec[i]=0 ;
		if (char_vec[i]=='A') vec[i]=0 ;
		if (char_vec[i]=='a') vec[i]=0 ;
		if (char_vec[i]=='C') vec[i]=1 ;
		if (char_vec[i]=='c') vec[i]=1 ;
		if (char_vec[i]=='G') vec[i]=2 ;
		if (char_vec[i]=='g') vec[i]=2 ;
		if (char_vec[i]=='T') vec[i]=3 ;
		if (char_vec[i]=='t') vec[i]=3 ;
		//assert(vec[i]!=-1) ;
	} ;
		
	for (INT i=0; i<len-degree; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		
		for (INT j=0; j<degree; j++)
		{
			tree->usage++ ;
			if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				tree=tree->childs[vec[i+j]] ;
			else
				break;
		} 
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
}

void CWeightedDegreeCharKernel::prune_tree(struct SuffixTree * p_tree, int min_usage)
{
	if (p_tree==NULL)
	{
		if (trees==NULL)
			return ;
		for (INT i=0; i<seq_length; i++)
			prune_tree(trees[i], min_usage) ;
		return ;
	}
	if (p_tree->has_floats)
		return ;
	
	for (INT i=0; i<4; i++)
	{
		if (p_tree->childs[i]!=NULL)
		{
			prune_tree(p_tree->childs[i], min_usage)  ;
			if (p_tree->childs[i]->usage < min_usage)
			{
				delete_tree(p_tree->childs[i]) ;
				delete p_tree->childs[i] ;
				p_tree->childs[i]=NULL ;
			}
		} 

	} ;
} 

void CWeightedDegreeCharKernel::delete_tree(struct SuffixTree * p_tree)
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


INT CWeightedDegreeCharKernel::tree_size(struct SuffixTree * p_tree)
{
	INT ret=0 ;

	if (p_tree==NULL)
	{
		if (trees==NULL)
			return 0 ;
		
		for (INT i=0; i<seq_length; i++)
			if (trees[i]!=NULL)
				ret += tree_size(trees[i]) ;
			else
				CIO::message(M_ERROR, "%i empty\n", i) ;
		return ret ;
	}
	if (p_tree->has_floats)
		return 4 ;

	for (INT i=0; i<4; i++)
		if (p_tree->childs[i]!=NULL)
			ret += tree_size(p_tree->childs[i])+1  ;

	return ret ;
} 

bool CWeightedDegreeCharKernel::set_weights(REAL* ws, INT d, INT len)
{
	degree=d;
	length=len;

	delete[] weights;
	weights=new REAL[d*len];

	if (weights)
	{
		for (int i=0; i<degree*length; i++)
			weights[i]=ws[i];
		return true;
	}
	else
		return false;
}
