#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "features/CombinedFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCombinedKernel::CCombinedKernel(LONG size)
	: CKernel(size)
{
	kernel_list=new CList<CKernel*>(true);
	fprintf(stderr, "combined kernel created\n") ;
	
}

CCombinedKernel::~CCombinedKernel() 
{
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
				fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X)\n", k, lf, rf) ;
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

				fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X): %i\n", k, lf, rf, result) ;

				lf=((CCombinedFeatures*) l)->get_next_feature_obj() ;
				k=get_next_kernel() ;
				rf=lf ;
			} ;
		}
	}
	fprintf(stderr,"k=0x%X  lf=0x%X  rf=0x%X  result=%i\n", k, lf, rf, result) ;

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
