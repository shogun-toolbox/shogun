#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "features/CombinedFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCombinedKernel::CCombinedKernel(LONG size)
	: CKernel(size)
{
	kernel_list=new CList<CKernel*>(true);
}

CCombinedKernel::~CCombinedKernel() 
{
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

	if ( (lf=((CCombinedFeatures*) l)->get_first_feature_obj()) &&
			(rf=((CCombinedFeatures*) r)->get_first_feature_obj()) && 
			(k=get_first_kernel()) )
	{
		result=k->init(lf,rf, do_init);

		while ( (result) && 
				(lf=((CCombinedFeatures*) l)->get_next_feature_obj()) &&
				(lf=((CCombinedFeatures*) r)->get_next_feature_obj()) &&
				(k=get_next_kernel()) )
			result=k->init(lf,rf, do_init);
	}

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
	CKernel* k=NULL;

	if ( (k=get_first_kernel()) )
	{
		k->cleanup();

		while ( (k=get_next_kernel()) )
			k->cleanup();
	}
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	CIO::message(M_INFO, "BEGIN COMBINED FEATURES LIST - ");
	this->list_kernel();

	if ( (k=get_first_kernel()) )
	{
		k->list_kernel();
		while ( (k=get_next_kernel()) )
			k->list_kernel();
	}
	CIO::message(M_INFO, "END COMBINED FEATURES LIST - ");
	this->list_kernel();
}

REAL CCombinedKernel::compute(INT x, INT y)
{
	REAL result=0;
	CKernel* k=get_first_kernel();
	while (k)
	{
		result+=k->kernel(x,y);
		k=get_next_kernel();
	}

	return result;
}
