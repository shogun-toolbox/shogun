#include "lib/common.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(LONG size, double* w, INT d, INT max_mismatch_, bool use_norm, bool block, INT mkl_stepsize_)
	: CCharKernel(size),weights(NULL),position_weights(NULL),weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), max_mismatch(max_mismatch_), seq_length(0),
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), match_vector(NULL), use_normalization(use_norm), block_computation(block)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION;
	lhs=NULL;
	rhs=NULL;
	matching_weights=NULL;

	weights=new REAL[d*(1+max_mismatch)];
	assert(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

#ifdef USE_TREEMEM
	TreeMemPtrMax=50000000 ;
	TreeMemPtr=0 ;
	TreeMem = new struct SuffixTree[TreeMemPtrMax] ;
#endif

	length = 0;
	trees=NULL;
	tree_initialized=false ;
}

CWeightedDegreeCharKernel::~CWeightedDegreeCharKernel() 
{
	cleanup();

	delete[] weights;
	weights=NULL;
	delete[] position_weights ;
	position_weights=NULL ;
	delete[] weights_buffer ;
	weights_buffer = NULL ;

#ifdef USE_TREEMEM
	delete[] TreeMem ;
#endif
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

	init_matching_weights_wd();

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
	delete[] matching_weights;
	matching_weights=NULL;

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
  

bool CWeightedDegreeCharKernel::init_optimization(INT count, INT * IDX, REAL * alphas)
{
	/*if (max_mismatch!=0)
	{
		CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not implemented for mismatch!=0\n") ;
		return false ;
		}*/

	delete_optimization();
	
	CIO::message(M_DEBUG, "initializing CWeightedDegreeCharKernel optimization\n") ;
	int i=0;
	for (i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::progress(i, 0, count);
		if (max_mismatch==0)
			add_example_to_tree(IDX[i], alphas[i]) ;
		else
			add_example_to_tree_mismatch(IDX[i], alphas[i]) ;
	}
	CIO::message(M_MESSAGEONLY, "done.           \n");
	
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


REAL CWeightedDegreeCharKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	REAL sum = 0.0 ;
	
	for (INT i=0; i<alen; i++)
	{
		REAL sumi = 0.0 ;
		INT mismatches=0 ;
		
		for (INT j=0; (i+j<alen) && (j<degree); j++)
		{
			if (match_vector[i+j])
			{
				mismatches++ ;
				if (mismatches>max_mismatch)
					break ;
			} ;
			sumi += weights[j+degree*mismatches];
		}
		if (position_weights!=NULL)
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

REAL CWeightedDegreeCharKernel::compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	REAL sum=0;

	INT match_len=-1;

	for (INT i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			match_len++;
		else
		{
			if (match_len>=0)
				sum+=matching_weights[match_len];
			match_len=-1;
		}
	}

	if (match_len>=0)
		sum+=matching_weights[match_len];

	return sum;
}

REAL CWeightedDegreeCharKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	REAL sum = 0.0 ;
	
	for (INT i=0; i<alen; i++)
	{
		REAL sumi = 0.0 ;
		
		for (INT j=0; (i+j<alen) && (j<degree); j++)
		{
			if (match_vector[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights!=NULL)
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

REAL CWeightedDegreeCharKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	REAL sum = 0.0 ;

	for (INT i=0; i<alen; i++)
	{
		REAL sumi=0.0 ;
		for (INT j=0; (i+j<alen) && (j<degree); j++)
		{
			if (match_vector[i+j])
				break;
			sumi += weights[i*degree+j];
		}
		if (position_weights!=NULL)
			sum += position_weights[i]*sumi ;
		else
			sum += sumi ;
	}

	return sum ;
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

  double result=0;

  if (max_mismatch == 0 && length == 0 && block_computation)
	  result = compute_using_block(avec, alen, bvec, blen) ;
  else
  {
	  for (INT i=0; i<alen; i++)
		  match_vector[i]=(avec[i]!=bvec[i]) ;

	  if (max_mismatch > 0)
		  result = compute_with_mismatch(avec, alen, bvec, blen) ;
	  else if (length==0)
		  result = compute_without_mismatch(avec, alen, bvec, blen) ;
	  else
		  result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;
  }
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) result/sqrt_both;
}

void CWeightedDegreeCharKernel::add_example_to_tree(INT idx, REAL alpha) 
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
		
	if (length == 0 || max_mismatch > 0)
	{
		for (INT i=0; i<len; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			REAL alpha_pw = alpha ;
			if (position_weights!=NULL)
				alpha_pw = alpha*position_weights[i] ;
			if (alpha_pw==0.0)
				continue ;
			for (INT j=0; (j<degree) && (i+j<len); j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					tree->weight += alpha_pw*weights[j];
				} else 
				{
					if ((j==degree-1) && (tree->has_floats))
					{
						tree->child_weights[vec[i+j]] += alpha_pw*weights[j];
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
							tree->child_weights[vec[i+j]] += alpha_pw*weights[j];
							break ;
						}
						else
						{
							assert(!tree->has_floats) ;
#ifdef USE_TREEMEM
							tree->childs[vec[i+j]]=&TreeMem[TreeMemPtr++];
							assert(TreeMemPtr<TreeMemPtrMax) ;
#elseif
							tree->childs[vec[i+j]]=new struct SuffixTree ;
							assert(tree->childs[vec[i+j]]!=NULL) ;
#endif
							tree=tree->childs[vec[i+j]] ;
							for (INT k=0; k<4; k++)
								tree->childs[k]=NULL ;
							tree->weight = alpha_pw*weights[j] ;
							tree->has_floats=false ;
							tree->usage=0 ;
						}
					}
				}
			}
		}
	}
	else
	{
		for (INT i=0; i<len; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			REAL alpha_pw = alpha ;
			if (position_weights!=NULL) 
				alpha_pw = alpha*position_weights[i] ;
			if (alpha_pw==0.0)
				continue ;
			INT max_depth = 0 ;
			for (INT j=0; (j<degree) && (i+j<len); j++)
				if (CMath::abs(weights[i*degree + j]*alpha_pw)>1e-8)
					max_depth = j+1 ;
			
			for (INT j=0; (j<max_depth) && (i+j<len); j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					tree->weight += alpha_pw*weights[i*degree + j];
				} else 
					if ((j==degree-1) && (tree->has_floats))
					{
						tree->child_weights[vec[i+j]] += alpha_pw*weights[i*degree + j];
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
							tree->child_weights[vec[i+j]] += alpha_pw*weights[i*degree + j];
							break ;
						}
						else
						{
							assert(!tree->has_floats) ;
#ifdef USE_TREEMEM
							tree->childs[vec[i+j]]=&TreeMem[TreeMemPtr++];
							assert(TreeMemPtr<TreeMemPtrMax) ;
#elseif
							tree->childs[vec[i+j]] = new struct SuffixTree ;
							assert(tree->childs[vec[i+j]]!=NULL) ;
#endif
							tree=tree->childs[vec[i+j]] ;
							for (INT k=0; k<4; k++)
								tree->childs[k]=NULL ;
							tree->weight = alpha_pw*weights[i*degree + j] ;
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

void CWeightedDegreeCharKernel::add_example_to_tree_mismatch(INT idx, REAL alpha) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	//assert(max_mismatch==0) ;
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
		REAL alpha_pw = alpha ;
		if (position_weights!=NULL)
			alpha_pw = alpha*position_weights[i] ;
		if (alpha_pw==0.0)
			continue ;
		add_example_to_tree_mismatch_recursion(tree, alpha_pw, &vec[i], len-i, 0, 0) ;
	}
	//fprintf(stdout,"*") ;

	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeCharKernel::add_example_to_tree_mismatch_recursion(struct SuffixTree *tree,  REAL alpha,
																	   INT *vec, INT len_rem, 
																	   INT degree_rec, INT mismatch_rec) 
{
	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	assert(tree!=NULL) ;
	const INT other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;
			
	struct SuffixTree *subtree = NULL ;

	if (degree_rec==degree-1)
	{
		if (tree->has_floats)
		{
			tree->child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
			if (mismatch_rec+1<=max_mismatch)
				for (INT o=0; o<3; o++)
					tree->child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
		}
		else
		{
			tree->has_floats=true ;
			for (INT k=0; k<4; k++)
			{
				assert(tree->childs[k]==NULL) ;
				tree->child_weights[k] =0 ;
			}
			tree->child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
			if (mismatch_rec+1<=max_mismatch)
				for (INT o=0; o<3; o++)
					tree->child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
		}
		return ;
	}
	else
	{
		assert(!tree->has_floats) ;
		if (tree->childs[vec[0]]!=NULL)
		{
			subtree=tree->childs[vec[0]] ;
			subtree->weight += alpha*weights[degree_rec+degree*mismatch_rec];
		} else 
		{
#ifdef USE_TREEMEM
			tree->childs[vec[0]]=&TreeMem[TreeMemPtr++] ;
			assert(TreeMemPtr<TreeMemPtrMax) ;
#else
			tree->childs[vec[0]]=new struct SuffixTree ;
			assert(tree->childs[vec[0]]!=NULL) ;
#endif			
			subtree=tree->childs[vec[0]] ;
			for (INT k=0; k<4; k++)
				subtree->childs[k]=NULL ;
			subtree->weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
			subtree->has_floats=false ;
			subtree->usage=0 ;
		}
		add_example_to_tree_mismatch_recursion(subtree,  alpha,
											   &vec[1], len_rem-1, 
											   degree_rec+1, mismatch_rec) ;

		if (mismatch_rec+1<=max_mismatch)
		{
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (tree->childs[ot]!=NULL)
				{
					subtree=tree->childs[ot] ;
					subtree->weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
				} else 
				{
#ifdef USE_TREEMEM
					tree->childs[ot]=&TreeMem[TreeMemPtr++] ;
					assert(TreeMemPtr<TreeMemPtrMax) ;
#else
					tree->childs[ot]=new struct SuffixTree ;
					assert(tree->childs[ot]!=NULL) ;
#endif
					subtree=tree->childs[ot] ;
					for (INT k=0; k<4; k++)
						subtree->childs[k]=NULL ;
					subtree->weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
					subtree->has_floats=false ;
					subtree->usage=0 ;
				}
				
				add_example_to_tree_mismatch_recursion(subtree,  alpha,
													   &vec[1], len_rem-1, 
													   degree_rec+1, mismatch_rec+1) ;
			}
		}
	}
}


REAL CWeightedDegreeCharKernel::compute_by_tree(INT idx) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	//assert(max_mismatch==0) ;
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
	for (INT i=0; i<len; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		
		for (INT j=0; (j<degree) && (i+j<len); j++)
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
	INT slen ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, slen, free);
	//assert(max_mismatch==0) ;
	INT *vec = new INT[slen] ;
	
	for (INT i=0; i<slen; i++)
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

	if (position_weights!=NULL)
	{
		for (INT i=0; i<slen; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			assert(tree!=NULL) ;
			
			for (INT j=0; (j<degree) && (i+j<slen); j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					LevelContrib[i/mkl_stepsize] += factor*tree->weight ;
				} else
					if (tree->has_floats)
					{
						LevelContrib[i/mkl_stepsize] += factor*tree->child_weights[vec[i+j]] ;
						break ;
					} else
						break ;
			} 
		}
	}
	else if (length==0)
	{
		//for (INT j=0; j<degree; j++)
		//LevelContrib[j]=0 ;
		for (INT i=0; i<slen; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			assert(tree!=NULL) ;
			
			for (INT j=0; (j<degree) && (i+j<slen); j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					LevelContrib[j/mkl_stepsize] += factor*tree->weight ;
				} else
					if (tree->has_floats)
					{
						LevelContrib[j/mkl_stepsize] += factor*tree->child_weights[vec[i+j]] ;
						break ;
					} else
						break ;
			} 
		}
	} 
	else
	{
		//for (INT j=0; j<degree*length; j++)
		//LevelContrib[j]=0 ;
		for (INT i=0; i<slen; i++)
		{
			struct SuffixTree *tree = trees[i] ;
			assert(tree!=NULL) ;
			
			for (INT j=0; (j<degree) && (i+j<slen); j++)
			{
				if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
				{
					tree=tree->childs[vec[i+j]] ;
					LevelContrib[(j+degree*i)/mkl_stepsize] += factor*tree->weight ;
				} else
					if (tree->has_floats)
					{
						LevelContrib[(j+degree*i)/mkl_stepsize] += factor*tree->child_weights[vec[i+j]] ;
						break ;
					} else
						break ;
			} 
		}
	} ;
	
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
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
	
	for (INT i=0; i<seq_length; i++)
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
	//assert(max_mismatch==0) ;
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
		
	for (INT i=0; i<len; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		assert(tree!=NULL) ;
		
		for (INT j=0; (j<degree) && (i+j<degree); j++)
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
#ifdef USE_TREEMEM
		TreeMemPtr=0;
#endif
		return;
	}

	if (p_tree->has_floats)
		return;
	
	for (INT i=0; i<4; i++)
	{
		if (p_tree->childs[i]!=NULL)
		{
#ifndef USE_TREEMEM
			delete_tree(p_tree->childs[i]);
			delete p_tree->childs[i];
#endif
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
	CIO::message(M_DEBUG, "degree = %i  d=%i\n", degree, d) ;
	degree = d ;
	length=len;
	
	if (len == 0)
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

bool CWeightedDegreeCharKernel::set_position_weights(REAL* pws, INT len)
{
	if (len==0)
	{
		delete[] position_weights ;
		position_weights = NULL ;
	}
	
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

bool CWeightedDegreeCharKernel::init_matching_weights_wd()
{
	delete[] matching_weights;
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		double deg=degree;
		INT k;
		for (k=0; k<degree ; k++)
			matching_weights[k]=(-pow(k,3) + (3*deg-3)*pow(k,2) + (9*deg-2)*k + 6*deg) / (3*deg*(deg+1));
		for (k=degree; k<seq_length ; k++)
			matching_weights[k]=(-deg+3*k+4)/3;
	}

	return (matching_weights!=NULL);
}
