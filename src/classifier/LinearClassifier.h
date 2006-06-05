#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "classifier/Classifier.h"

#include <stdio.h>

class CLinearClassifier : public CClassifier
{
	public:
		CLinearClassifier();
		virtual ~CLinearClassifier();

		/// get output for example "idx"
		virtual inline DREAL classify_example(INT idx)
		{
			INT vlen;
			bool vfree;
			double* vec=features->get_feature_vector(idx, vlen, vfree);
			DREAL result=CMath::dot(w,vec,vlen);
			features->free_feature_vector(vec, idx, vfree);

			return result+bias;
		}

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		virtual inline void set_features(CRealFeatures* feat) { features=feat; }
		virtual CRealFeatures* get_features() { return features; }

	protected:
		DREAL* w;
		DREAL bias;
		CRealFeatures* features;
};
#endif
