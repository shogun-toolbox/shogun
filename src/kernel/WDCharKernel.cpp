#include "lib/common.h"
#include "kernel/WDCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWDCharKernel::CWDCharKernel(LONG size, EWDKernType t, INT d)
	: CCharKernel(size), type(t), degree(d), seq_length(0),
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), match_vector(NULL)
{
	matching_weights=NULL; //depend on length of sequence will be initialized later

	lhs=NULL ;
	rhs=NULL ;
}

CWDCharKernel::~CWDCharKernel() 
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
	delete[] match_vector ;

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

bool CWDCharKernel::init_matching_weights_wd()
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

bool CWDCharKernel::init_matching_weights_const()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<seq_length+1 ; i++)
			matching_weights[i-1]=1.0/seq_length;
	}

	return (matching_weights!=NULL);
}

bool CWDCharKernel::init_matching_weights_linear()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<seq_length+1 ; i++)
			matching_weights[i-1]=degree*i;
	}

	return (matching_weights!=NULL);
}

bool CWDCharKernel::init_matching_weights_sqpoly()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			matching_weights[i-1]=((double) i)*i;

		for (int i=degree+1; i<seq_length+1 ; i++)
			matching_weights[i-1]=i;
	}

	return (matching_weights!=NULL);
}

bool CWDCharKernel::init_matching_weights_cubicpoly()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			matching_weights[i-1]=((double) i)*i*i;

		for (int i=degree+1; i<seq_length+1 ; i++)
			matching_weights[i-1]=i;
	}

	return (matching_weights!=NULL);
}

bool CWDCharKernel::init_matching_weights_exp()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			matching_weights[i-1]=exp(((double) i/10.0));

		for (int i=degree+1; i<seq_length+1 ; i++)
			matching_weights[i-1]=i;
	}

	return (matching_weights!=NULL);
}

bool CWDCharKernel::init_matching_weights_log()
{
	matching_weights=new REAL[seq_length];

	if (matching_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			matching_weights[i-1]=pow(log(i),2);

		for (int i=degree+1; i<seq_length+1 ; i++)
			matching_weights[i-1]=i;
	}

	return (matching_weights!=NULL);
}


bool CWDCharKernel::init_matching_weights()
{
	switch (type)
	{
		case E_WD:
			return init_matching_weights_wd();
		case E_CONST:
			return init_matching_weights_const();
		case E_LINEAR:
			return init_matching_weights_linear();
		case E_SQPOLY:
			return init_matching_weights_sqpoly();
		case E_CUBICPOLY:
			return init_matching_weights_cubicpoly();
		case E_EXP:
			return init_matching_weights_exp();
		case E_LOG:
			return init_matching_weights_log();
		default:
			return false;
	};

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
	matching_weights=NULL;
}

bool CWDCharKernel::load_init(FILE* src)
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
		matching_weights[i]=w[i];

    CIO::message(M_INFO, "detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	return true;
}

bool CWDCharKernel::save_init(FILE* dest)
{
	return false;
}
  

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

	((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return (double) sum/sqrt_both;
}
