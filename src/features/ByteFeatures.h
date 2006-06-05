#ifndef _BYTEFEATURES__H__
#define _BYTEFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CByteFeatures: public CSimpleFeatures<BYTE>
{
	public:
		CByteFeatures(LONG size);
		CByteFeatures(const CByteFeatures & orig);
		CByteFeatures(CHAR* fname);

		virtual EFeatureType get_feature_type() { return F_BYTE; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
};
#endif
