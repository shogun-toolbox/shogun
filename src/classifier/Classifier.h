#ifndef _CLASSIFIER_H__
#define _CLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"

#include <stdio.h>

class CClassifier
{
	public:
		CClassifier();
		virtual ~CClassifier();

		virtual bool train()=0;
		virtual REAL* test();

		virtual REAL classify_example(INT num)=0;

		virtual bool load(FILE* srcfile)=0;
		virtual bool save(FILE* dstfile)=0;

		virtual inline void set_labels(CLabels* lab) { labels=lab; }
		virtual inline CLabels* get_labels() { return labels; }

	protected:
		CLabels* labels;
};
#endif

