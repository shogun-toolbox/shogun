#ifndef _CHARFEATURES__H__
#define _CHARFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CCharFeatures: public CSimpleFeatures<CHAR>
{
	public:
		CCharFeatures(E_OBS_ALPHABET alphabet, long size);
		CCharFeatures(const CCharFeatures & orig);
		CCharFeatures(char* fname);

		/// remap element e.g translate ACGT to 0123
		inline CHAR remap(CHAR c)
		{
			return maptable[c];
		}

		virtual EFeatureType get_feature_type() { return F_CHAR; }

		virtual CFeatures* duplicate() const;
		virtual bool load(char* fname);
		virtual bool save(char* fname);
	public:
		static const unsigned char B_A;
		static const unsigned char B_C;
		static const unsigned char B_G;
		static const unsigned char B_T;
		static const unsigned char B_N;
		static const unsigned char B_n;
	protected:
		void init_map_table();
		CHAR maptable[1 << (sizeof(CHAR)*8)];
		E_OBS_ALPHABET alphabet_type;
};
#endif
