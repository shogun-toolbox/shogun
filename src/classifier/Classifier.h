/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CLASSIFIER_H__
#define _CLASSIFIER_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Mathematics.h"
#include "features/Labels.h"

class CClassifier : public CSGObject
{
	public:
		CClassifier();
		virtual ~CClassifier();

		virtual bool train() { return false; }
		virtual CLabels* classify(CLabels* output=NULL);

		virtual DREAL classify_example(INT num) { return CMath::INFTY; }

		virtual bool load(FILE* srcfile) { ASSERT(srcfile); return false; }
		virtual bool save(FILE* dstfile) { ASSERT(dstfile); return false; }

		virtual inline void set_labels(CLabels* lab) { SG_REF(lab);; labels=lab; }
		virtual inline CLabels* get_labels() { SG_REF(labels); return labels; }
		virtual inline DREAL get_label(INT i) { return labels->get_label(i); }

		virtual EClassifierType get_classifier_type() { return CT_NONE; }

	protected:
		CLabels* labels;
};
#endif
