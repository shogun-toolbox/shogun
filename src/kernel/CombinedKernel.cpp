#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "features/CombinedFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCombinedKernel::CCombinedKernel(LONG size)
	: CKernel(size), sv_count(0), sv_idx(NULL), sv_weight(NULL), subkernel_weights_buffer(NULL)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION;
	kernel_list=new CList<CKernel*>(true);
	fprintf(stderr, "combined kernel created\n") ;
}

CCombinedKernel::~CCombinedKernel() 
{
	delete[] subkernel_weights_buffer ;
	subkernel_weights_buffer=NULL ;
	
	CIO::message(M_INFO, "combined kernel deleted\n");
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
	delete_optimization();

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

	delete_optimization();
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	CIO::message(M_INFO, "BEGIN COMBINED KERNEL LIST - ");
	this->list_kernel();

	k=get_first_kernel();
	while (k)
	{
		k->list_kernel();
		k=get_next_kernel();
	}
	CIO::message(M_INFO, "END COMBINED KERNEL LIST - ");
}

REAL CCombinedKernel::compute(INT x, INT y)
{
	REAL result=0;
	CKernel* k=get_first_kernel();
	while (k)
	{
		if (k->get_combined_kernel_weight()!=NULL)
			result += k->get_combined_kernel_weight() * k->kernel(x,y);
		k=get_next_kernel();
	}

	return result;
}

bool CCombinedKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	CIO::message(M_DEBUG, "initializing CCombinedKernel optimization\n") ;

	delete_optimization();
	
	CKernel * k = get_first_kernel() ;
	bool have_non_optimizable = false ;
	
	while(k)
	{
		bool ret = true ;
		
		if (k && k->is_optimizable())
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

bool CCombinedKernel::delete_optimization() 
{ 
	//CIO::message(M_DEBUG, "deleting CCombinedKernel optimization\n") ;

	CKernel* k = get_first_kernel();

	while(k)
	{
		k->delete_optimization();
		k = get_next_kernel();
	}

	delete[] sv_idx;
	sv_idx = NULL;

	delete[] sv_weight;
	sv_weight = NULL;

	sv_count = 0;
	set_is_initialized(false);

	return true;
}

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
		if (k && k->is_optimizable() && 
			k->get_is_initialized())
		{
			if (k->get_combined_kernel_weight()!=0)
				result += k->get_combined_kernel_weight()*
					k->compute_optimized(idx) ;
		}
		else
		{
			assert(sv_idx!=NULL || sv_count==0) ;
			assert(sv_weight!=NULL || sv_count==0) ;
			// CIO::message(M_DEBUG, "not optimized kernel computation\n") ;

			if (k->get_combined_kernel_weight()!=0)
			{ // compute the usual way for any non-optimized kernel
				int j=0;
				REAL sub_result=0 ;
				for (j=0; j<sv_count; j++)
					sub_result += sv_weight[j] * k->kernel(sv_idx[j], idx) ;
				
				result += k->get_combined_kernel_weight()*sub_result ;
			}
		} ;
		
		k = get_next_kernel() ;
	}

	return result;
}

void CCombinedKernel::add_to_normal(INT idx, REAL weight) 
{ 
	if (!get_is_initialized())
	{
		init_optimization(1, &idx, &weight) ;
		return ;
	}

	CKernel* k = get_first_kernel();

	while(k)
	{
		k->add_to_normal(idx, weight);
		k = get_next_kernel();
	}
}

void CCombinedKernel::clear_normal() 
{ 
	CKernel* k = get_first_kernel();

	while(k)
	{
		k->clear_normal() ;
		k = get_next_kernel();
	}
}

void CCombinedKernel::compute_by_subkernel(INT idx, REAL * subkernel_contrib)
{
	INT i=0 ;
	CKernel* k = get_first_kernel();
	while(k)
	{
		if (k->get_combined_kernel_weight()!=0)
			subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;
		//subkernel_contrib[i] += k->compute_optimized(idx) ;
		k = get_next_kernel();
		i++ ;
	}
}

const REAL* CCombinedKernel::get_subkernel_weights(INT& num_weights)
{
	num_weights = kernel_list->get_num_elements() ;
	delete[] subkernel_weights_buffer ;
	subkernel_weights_buffer = new REAL[num_weights] ;

	INT i=0 ;
	CKernel* k = get_first_kernel();
	while(k)
	{
		subkernel_weights_buffer[i] = k->get_combined_kernel_weight();
		//CIO::message(M_INFO, "g %i: %1.1f ", i, subkernel_weights_buffer[i]) ;
		k = get_next_kernel();
		i++ ;
	}
	return subkernel_weights_buffer ;
}

void CCombinedKernel::set_subkernel_weights(REAL* weights, INT num_weights)
{
	INT i=0 ;
	CKernel* k = get_first_kernel();
	while(k)
	{
		k->set_combined_kernel_weight(weights[i]);
		//CIO::message(M_INFO, "s %i: %1.1f ", i, weights[i]) ;
		k = get_next_kernel();
		i++ ;
	}
}
