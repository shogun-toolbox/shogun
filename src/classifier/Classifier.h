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
#include "lib/Mathematics.h"
#include "features/Labels.h"

#include <stdio.h>
#include <cstring>
using namespace std;

class ClassifierException {
      private:
         char *mes;
      public:
         ClassifierException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};


class CClassifier
{
	public:
		CClassifier();
		virtual ~CClassifier();

		virtual bool train() { return false; }
		virtual CLabels* classify(CLabels* output=NULL);

		virtual DREAL classify_example(INT num) { return CMath::INFTY; }

		virtual bool load(FILE* srcfile) { return false; }
		virtual bool save(FILE* dstfile) { return false; }

		virtual inline void set_labels(CLabels* lab) { labels=lab; }
		virtual inline CLabels* get_labels() { return labels; }

		virtual EClassifierType get_classifier_type() { return CT_NONE; }

	protected:
		CLabels* labels;
};
#endif

