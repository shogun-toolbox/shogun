#include "lib/common.h"
#include "kernel/WDCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWDCharKernel::CWDCharKernel(LONG size, INT d, INT max_mismatch_)
	: CCharKernel(size),degree(d), max_mismatch(max_mismatch_), seq_length(0),
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), match_vector(NULL)
{
	old_weights=new REAL[d*(1+max_mismatch)];
	matching_weights=NULL; //depend on length of sequence will be initialized later

	INT i=0;
	REAL sum=0;

	for (i=0; i<d; i++)
	{
		old_weights[i]=d-i;
		sum+=old_weights[i];
	}

	for (i=0; i<d; i++)
		old_weights[i]/=sum;

	for (i=0; i<d; i++)
	{
		for (INT j=1; j<=max_mismatch; j++)
		{
			if (j<i+1)
			{
				INT nk=math.nchoosek(i+1, j);
				old_weights[i+j*d]=old_weights[i]/(nk*pow(3,j));
			}
			else
				old_weights[i+j*d]= 0;
		}
	}

	lhs=NULL ;
	rhs=NULL ;

	trees=NULL ;
	tree_initialized=false ;
}

CWDCharKernel::~CWDCharKernel() 
{
	delete_optimization() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
	delete[] match_vector ;

	if (trees!=NULL)
	{
		for (INT i=0; i<seq_length; i++)
		{
			delete trees[i] ;
			trees[i]=NULL ;
		}
		delete[] trees ;
		trees=NULL ;
	}
	cleanup();
}

void CWDCharKernel::remove_lhs() 
{ 
	if (get_is_initialized())
		delete_optimization() ;
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
			delete trees[i] ;
			trees[i]=NULL ;
		}
		delete[] trees ;
		trees=NULL ;
	}
}

void CWDCharKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}


bool CWDCharKernel::init_matching_weights()
{

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

bool CWDCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
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

		init_matching_weights();
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
	
	this->lhs=(CCharFeatures*) l;
	this->rhs=(CCharFeatures*) r;

	initialized = true ;
	return result;
}
void CWDCharKernel::cleanup()
{
	delete[] old_weights;
	old_weights=NULL;
}

bool CWDCharKernel::load_init(FILE* src)
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
	double* old_weights= new double[d];
	assert(old_weights) ;
	
    assert(fread(w, sizeof(double), d, src)==(UINT) d) ;

	for (INT i=0; i<d; i++)
		old_weights[i]=w[i];

    CIO::message(M_INFO, "detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	return true;
}

bool CWDCharKernel::save_init(FILE* dest)
{
	return false;
}
  

bool CWDCharKernel::init_optimization(INT count, INT * IDX, REAL * old_weights)
{
	if (max_mismatch!=0)
	{
		CIO::message(M_ERROR, "CWDCharKernel optimization not implemented for mismatch!=0\n") ;
		return false ;
	}

	if (get_is_initialized()) 
		delete_optimization() ;
	
	CIO::message(M_DEBUG, "initializing CWDCharKernel optimization\n") ;
	int i=0;
	for (i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::message(M_PROGRESS, "%3i%%  \r", 100*i/(count+1)) ;
		add_example_to_tree(IDX[i], old_weights[i]) ;
	}
	CIO::message(M_PROGRESS, "done.           \n");
	
	set_is_initialized(true) ;
	return true ;
}

void CWDCharKernel::delete_optimization() 
{ 
	if (get_is_initialized())
	{
		CIO::message(M_DEBUG, "deleting CWDCharKernel optimization\n") ;
		
		delete_tree(NULL); 
		set_is_initialized(false) ;
	} else
		CIO::message(M_ERROR, "CWDCharKernel optimization not initialized\n") ;
} ;

REAL CWDCharKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	// can only deal with strings of same length
	assert(alen==blen);

	REAL sqrt_a= 1 ;
	REAL sqrt_b= 1 ;
	if (initialized)
	{
		sqrt_a=sqrtdiag_lhs[idx_a] ;
		sqrt_b=sqrtdiag_rhs[idx_b] ;
	}

	REAL sqrt_both=sqrt_a*sqrt_b;

	REAL sum=0;

	if (max_mismatch==0)
	{
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
	}
	else
		CIO::message(M_ERROR, "mismatches not supported\n");

	((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return (double) sum/sqrt_both;
}

void CWDCharKernel::add_example_to_tree(INT idx, REAL weight) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	assert(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	weight /= sqrtdiag_lhs[idx] ;
	
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
		
	for (INT i=0; i<len-degree; i++)
	{
		struct SuffixTree *tree = trees[i] ;
		for (INT j=0; j<degree; j++)
		{
			if ((!tree->has_floats) && (tree->childs[vec[i+j]]!=NULL))
			{
				tree=tree->childs[vec[i+j]] ;
				tree->weight += weight*old_weights[j];
			} else 
				if ((j==degree-1) && (tree->has_floats))
				{
					tree->child_weights[vec[i+j]] += weight*old_weights[j];
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
						tree->child_weights[vec[i+j]] += weight*old_weights[j];
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
						tree->weight = weight*old_weights[j] ;
						tree->has_floats=false ;
						tree->usage=0 ;
					} ;
				} ;
		} ;
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

REAL CWDCharKernel::compute_by_tree(INT idx) 
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

	return sum/sqrtdiag_rhs[idx] ;
}

REAL CWDCharKernel::compute_abs_weights_tree(struct SuffixTree* p_tree) 
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

REAL *CWDCharKernel::compute_abs_weights(int &len) 
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

void CWDCharKernel::count_tree_usage(INT idx) 
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

void CWDCharKernel::prune_tree(struct SuffixTree * p_tree, int min_usage)
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

void CWDCharKernel::delete_tree(struct SuffixTree * p_tree)
{
	if (p_tree==NULL)
	{
		if (trees==NULL)
			return ;
		for (INT i=0; i<seq_length; i++)
		{
			delete_tree(trees[i]) ;
			trees[i]->has_floats=false ;
			trees[i]->usage=0;
			for (INT k=0; k<4; k++)
				trees[i]->childs[k]=NULL ;
		} ;		

		tree_initialized=false ;
		return ;
	}
	if (p_tree->has_floats)
		return ;
	
	for (INT i=0; i<4; i++)
	{
		if (p_tree->childs[i]!=NULL)
		{
			delete_tree(p_tree->childs[i])  ;
			delete p_tree->childs[i] ;
			p_tree->childs[i]=NULL ;
		} 
		p_tree->weight=0 ;
	}
} 


INT CWDCharKernel::tree_size(struct SuffixTree * p_tree)
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
