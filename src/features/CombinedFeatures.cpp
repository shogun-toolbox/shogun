#include "features/CombinedFeatures.h"
#include "lib/io.h"

class CCombinedFeatures;

CCombinedFeatures::CCombinedFeatures() : CFeatures(0l)
{
	feature_list=new CList<CFeatures*>(true);
}

CCombinedFeatures::CCombinedFeatures(const CCombinedFeatures & orig) : CFeatures(0l)
{
}

CFeatures* CCombinedFeatures::duplicate() const
{
	return new CCombinedFeatures(*this);
}

CCombinedFeatures::~CCombinedFeatures()
{
	delete feature_list;
}

void CCombinedFeatures::list_feature_objs()
{
	CFeatures* f;

	CIO::message(M_INFO, "BEGIN COMBINED FEATURES LIST - ");
	this->list_feature_obj();

	CListElement<CFeatures*> * current = NULL ;
	f=get_first_feature_obj(current);

	while (f)
	{
		f->list_feature_obj();
		f=get_next_feature_obj(current);
	}

	CIO::message(M_INFO, "END COMBINED FEATURES LIST - ");
}

bool CCombinedFeatures::check_feature_obj_compatibility(CCombinedFeatures* comb_feat)
{
	bool result=false;

	if (comb_feat && (this->get_num_feature_obj() == comb_feat->get_num_feature_obj()) )
	{
		CFeatures* f1=this->get_first_feature_obj();
		CFeatures* f2=comb_feat->get_first_feature_obj();

		if (f1 && f1->check_feature_compatibility(f2))
		{
			while( ( (f1=this->get_first_feature_obj()) != NULL )  && 
				   ( (f2=comb_feat->get_first_feature_obj()) != NULL) )
			{
				if (!f1->check_feature_compatibility(f2))
					return false;
			}
			result=true;
		}
	}

	return result;
}
