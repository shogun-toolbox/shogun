#ifndef _SHORTFEATURES__H__
#define _SHORTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CShortFeatures: public CSimpleFeatures<SHORT>
{
	public:
		CShortFeatures(long size);
		CShortFeatures(const CShortFeatures & orig);

		/** load features from file
		 * fname - filename
		 */

		CShortFeatures(char* fname);

		bool obtain_from_char_features(CCharFeatures* cf, E_OBS_ALPHABET alphabet, int start, int order);

		virtual EType get_feature_type() { return F_SHORT; }

		virtual bool load(char* fname);
		virtual bool save(char* fname);
	protected:
		void translate_from_single_order(SHORT* obs, int sequence_length, int start, int order, int max_val);

};
#endif
