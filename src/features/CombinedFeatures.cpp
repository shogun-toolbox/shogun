#include "features/CombinedFeatures.h"

class CCombinedFeatures;

CCombinedFeatures::CCombinedFeatures() : CFeatures(0l), feature_list(NULL)
{
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
}
