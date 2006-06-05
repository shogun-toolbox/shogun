#ifndef _CHARFEATURES__H__
#define _CHARFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CCharFeatures: public CSimpleFeatures<CHAR>
{
	public:
		CCharFeatures(E_ALPHABET alphabet, LONG size);
		CCharFeatures(const CCharFeatures & orig);
		CCharFeatures(E_ALPHABET alphabet, CHAR* fname);

		/// remap element e.g translate ACGT to 0123
		inline CHAR remap(CHAR c)
		{
			return maptable[ (BYTE) c];
		}

		inline E_ALPHABET get_alphabet()
		{
			return alphabet_type;
		}

		virtual EFeatureType get_feature_type() { return F_CHAR; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	public:
		static const BYTE B_A;
		static const BYTE B_C;
		static const BYTE B_G;
		static const BYTE B_T;
		static const BYTE B_star;
		static const BYTE B_N;
		static const BYTE B_n;
	protected:
		void init_map_table();
		CHAR maptable[1 << (sizeof(CHAR)*8)];
		E_ALPHABET alphabet_type;
};
#endif
