#ifndef _SHORTFEATURES__H__
#define _SHORTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CShortFeatures: public CSimpleFeatures<SHORT>
{
	public:
		CShortFeatures(LONG size);
		CShortFeatures(const CShortFeatures & orig);

		/** load features from file
		 * fname - filename
		 */

		CShortFeatures(CHAR* fname);

		bool obtain_from_char_features(CCharFeatures* cf, INT start, INT order, INT gap=0);

		virtual EFeatureType get_feature_type() { return F_SHORT; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	protected:
		void translate_from_single_order(SHORT* obs, INT sequence_length, INT start, INT order, INT max_val, INT gap);

};
#endif
