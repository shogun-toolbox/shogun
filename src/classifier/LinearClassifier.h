#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"

#include <stdio.h>

class CLinearClassifier
{
	public:
		CLinearClassifier();
		virtual ~CLinearClassifier();

		virtual bool	train()=0;
		virtual REAL*	test();
		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

	protected:
		REAL* w;
		REAL bias;
};
#endif
