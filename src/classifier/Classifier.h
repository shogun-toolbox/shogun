/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CLASSIFIER_H__
#define _CLASSIFIER_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Mathematics.h"
#include "features/Labels.h"

/** class Classifier */
class CClassifier : public CSGObject
{
	public:
		/** constructor */
		CClassifier();
		virtual ~CClassifier();

		/** train classifier
		 *
		 * @return whether training was successful
		 */
		virtual bool train() { return false; }

		/** classify object
		 *
		 * @param output classified labels
		 * @return classified labels
		 */
		virtual CLabels* classify(CLabels* output=NULL);

		/** classify one example
		 *
		 * abstract base method
		 *
		 * @param num which example to classify
		 * @return infinite float value
		 */
		virtual DREAL classify_example(INT num) { return CMath::INFTY; }

		/** load Classifier from file
		 *
		 * abstract base method
		 *
		 * @param srcfile file to load from
		 * @return failure
		 */
		virtual bool load(FILE* srcfile) { ASSERT(srcfile); return false; }

		/** save Classifier to file
		 *
		 * abstract base method
		 *
		 * @param dstfile file to save to
		 * @return failure
		 */
		virtual bool save(FILE* dstfile) { ASSERT(dstfile); return false; }

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual inline void set_labels(CLabels* lab)
		{
			SG_UNREF(labels);
			SG_REF(lab);
			labels=lab;
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual inline CLabels* get_labels() { SG_REF(labels); return labels; }

		/** get one specific label
		 *
		 * @param i index of label to get
		 * @return value of label at index i
		 */
		virtual inline DREAL get_label(INT i) { return labels->get_label(i); }

		/** set maximum training time
		 *
		 * @param t maximimum training time
		 */
		inline void set_max_train_time(DREAL t) { max_train_time=t; }

		/** get maximum training time
		 *
		 * @return maximum training time
		 */
		inline DREAL get_max_train_time() { return max_train_time; }

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_NONE; }

	protected:
		/** maximum training time */
		DREAL max_train_time;

		/** labels */
		CLabels* labels;
};
#endif
