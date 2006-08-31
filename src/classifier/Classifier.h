/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
		virtual CLabels* classify(CLabels* output=NULL);

		virtual DREAL classify_example(INT num)=0;

		virtual bool load(FILE* srcfile)=0;
		virtual bool save(FILE* dstfile)=0;

		virtual inline void set_labels(CLabels* lab) { labels=lab; }
		virtual inline CLabels* get_labels() { return labels; }

		virtual EClassifierType get_classifier_type()=0;

	protected:
		CLabels* labels;
};
#endif

