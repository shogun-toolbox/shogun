#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include "features/Features.h"

class CFeatures;
class CCombinedFeatures;

class CCombinedFeatures : public CFeatures
{
public:
	CCombinedFeatures();
	CCombinedFeatures(const CCombinedFeatures& orig);
	virtual CFeatures* duplicate() const;
	virtual ~CCombinedFeatures();

	virtual EFeatureType get_feature_type() { return F_UNKNOWN; }
	virtual EFeatureClass get_feature_class() { return C_COMBINED; }
	virtual INT get_num_vectors() { return 0; }
	virtual INT get_size() { return 0; }

	protected:
		CFeatures* feature_list;
};
#endif

