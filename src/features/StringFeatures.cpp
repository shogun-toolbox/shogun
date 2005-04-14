#include "features/StringFeatures.h"
#include "lib/common.h"

inline EFeatureType CStringFeatures<REAL>::get_feature_type()
{
	return F_REAL;
}

inline EFeatureType CStringFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

inline EFeatureType CStringFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

inline EFeatureType CStringFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

inline EFeatureType CStringFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

inline EFeatureType CStringFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}
