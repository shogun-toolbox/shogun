#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "features/CombinedFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCombinedKernel::CCombinedKernel(LONG size)
	: CKernel(size), sv_count(0), sv_idx(NULL), sv_weight(NULL) 
{
	kernel_list=new CList<CKernel*>(true);
	fprintf(stderr, "combined kernel created\n") ;
	
}

CCombinedKernel::~CCombinedKernel() 
{
	if (get_is_initialized())
		delete_optimization() ;
	
	fprintf(stderr, "combined kernel deleted\n") ;
	cleanup();
	delete kernel_list;
}

bool CCombinedKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l,r, do_init);
	assert(l->get_feature_class() == C_COMBINED);
	assert(r->get_feature_class() == C_COMBINED);
	assert(l->get_feature_type() == F_UNKNOWN);
	assert(r->get_feature_type() == F_UNKNOWN);

	CFeatures* lf=NULL;
	CFeatures* rf=NULL;
	CKernel* k=NULL;

	bool result=true;

	lf=((CCombinedFeatures*) l)->get_first_feature_obj() ;
	rf=((CCombinedFeatures*) r)->get_first_feature_obj() ;
	k=get_first_kernel() ;

	result = 1 ;
	
	if ( lf && rf && k)
	{
		if (l!=r)
		{
			while ( result && lf && rf && k )
			{
				//fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X)\n", k, lf, rf) ;
				result=k->init(lf,rf, do_init);

				lf=((CCombinedFeatures*) l)->get_next_feature_obj() ;
				rf=((CCombinedFeatures*) r)->get_next_feature_obj() ;
				k=get_next_kernel() ;
			} ;
		}
		else
		{
			while ( result && lf && k )
			{
				result=k->init(lf,rf, do_init);

				//fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X): %i\n", k, lf, rf, result) ;

				lf=((CCombinedFeatures*) l)->get_next_feature_obj() ;
				k=get_next_kernel() ;
				rf=lf ;
			} ;
		}
	}
	//fprintf(stderr,"k=0x%X  lf=0x%X  rf=0x%X  result=%i\n", k, lf, rf, result) ;

	if (!result)
	{
		CIO::message(M_INFO, "CombinedKernel: Initialising the following kernel failed\n");
		if (k)
			k->list_kernel();
		else
			CIO::message(M_INFO, "<NULL>\n");
		return false;
	}

	if ((lf!=NULL) || (rf!=NULL) || (k!=NULL))
	{
		CIO::message(M_INFO, "CombinedKernel: Number of features/kernels does not match - bailing out\n");
		return false;
	}
	
	return true;
}

void CCombinedKernel::remove_lhs()
{
	if (get_is_initialized())
		delete_optimization() ;

	if (lhs)
		cache_reset() ;
	lhs=NULL ;
	
	CKernel* k=get_first_kernel();

	while (k)
	{	
		k->remove_lhs();
		k=get_next_kernel();
	}
}

void CCombinedKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;
	rhs=NULL ;

	CKernel* k=get_first_kernel();

	while (k)
	{	
		k->remove_rhs();
		k=get_next_kernel();
	}
}

void CCombinedKernel::cleanup()
{
	CKernel* k=get_first_kernel();

	while (k)
	{	
		k->cleanup();
		k=get_next_kernel();
	}
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	CIO::message(M_INFO, "BEGIN COMBINED FEATURES LIST - ");
	this->list_kernel();

	k=get_first_kernel();
	while (k)
	{
		k->list_kernel();
		k=get_next_kernel();
	}
	CIO::message(M_INFO, "END COMBINED FEATURES LIST - ");
}

REAL CCombinedKernel::compute(INT x, INT y)
{
	REAL result=0;
	CKernel* k=get_first_kernel();
	while (k)
	{
		result += k->get_combined_kernel_weight() * k->kernel(x,y);
		k=get_next_kernel();
	}

	return result;
}

bool CCombinedKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	CIO::message(M_DEBUG, "initializing CCombinedKernel optimization\n") ;

	if (get_is_initialized())
		delete_optimization() ;
	
	CKernel * k = get_first_kernel() ;
	bool have_non_optimizable = false ;
	
	while(k)
	{
		bool ret = true ;
		
		if (COptimizableKernel::is_optimizable(k))
			ret = k->init_optimization(count, IDX, weights) ;
		else
		{
			CIO::message(M_WARN, "non-optimizable kernel 0x%X in kernel-list\n",k) ;
			have_non_optimizable = true ;
		}
		
		if (!ret)
		{
			have_non_optimizable = true ;
			CIO::message(M_WARN, "init_optimization of kernel 0x%X failed\n",k) ;
		} ;
		
		k = get_next_kernel() ;
	}
	
	if (have_non_optimizable)
	{
		CIO::message(M_WARN, "some kernels in the kernel-list are not optimized\n") ;

		sv_idx = new INT[count] ;
		sv_weight = new REAL[count] ;
		sv_count = count ;
		int i ;
		for (i=0; i<count; i++)
		{
			sv_idx[i] = IDX[i] ;
			sv_weight[i] = weights[i] ;
		}
	}
	set_is_initialized(true) ;
	
	return true ;
} ;

void CCombinedKernel::delete_optimization() 
{ 
	CIO::message(M_DEBUG, "deleting CCombinedKernel optimization\n") ;

	CKernel * k = get_first_kernel() ;
	while(k)
	{
		if (COptimizableKernel::is_optimizable(k) && k->get_is_initialized())
			k->delete_optimization() ;
		k = get_next_kernel() ;
	}
	delete[] sv_idx ;
	delete[] sv_weight ;
	sv_count = 0 ;
	
	set_is_initialized(false) ;
} ;

REAL CCombinedKernel::compute_optimized(INT idx) 
{ 		  
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCombinedKernel optimization not initialized\n") ;
		return 0 ;
	}
	
	REAL result = 0 ;
	
	CKernel * k = get_first_kernel() ;
	while(k)
	{
		if (COptimizableKernel::is_optimizable(k) && 
			k->get_is_initialized())
			result += k->get_combined_kernel_weight()*
				k->compute_optimized(idx) ;
		else
		{
			assert(sv_idx!=NULL || sv_count==0) ;
			assert(sv_weight!=NULL || sv_count==0) ;
			CIO::message(M_DEBUG, "not optimized kernel computation\n") ;

			// compute the usual way for any non-optimized kernel
			int j=0;
			REAL sub_result=0 ;
			for (j=0; j<sv_count; j++)
				sub_result += sv_weight[j] * k->kernel(sv_idx[j], idx) ;
			
			result += k->get_combined_kernel_weight()*sub_result ;
		} ;
		
		k = get_next_kernel() ;
	}
	return result ; 
} ;
