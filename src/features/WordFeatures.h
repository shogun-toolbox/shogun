#ifndef _WORDFEATURES__H__
#define _WORDFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CWordFeatures: public CSimpleFeatures<WORD>
{
	public:
		CWordFeatures(LONG size, INT num_symbols=1<<16);
		CWordFeatures(const CWordFeatures & orig);

		/** load features from file
		 * fname - filename
		 */

		CWordFeatures(CHAR* fname, INT num_symbols=1<<16);

		bool obtain_from_char_features(CCharFeatures* cf, E_ALPHABET alphabet, INT start, INT order);

		virtual EFeatureType get_feature_type() { return F_WORD; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);

		inline INT get_num_symbols() { return num_symbols; }

	protected:
		///max_val is how many bits does the largest symbol require to be stored without loss
		void translate_from_single_order(WORD* obs, INT sequence_length, INT start, INT order, INT max_val);

	protected:
		INT num_symbols;

};
#endif
