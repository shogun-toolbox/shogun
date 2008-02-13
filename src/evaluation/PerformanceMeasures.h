/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Sebastian Henschel
 * Copyright (C) 2008 Friedrich Miescher Laboratory of Max-Planck-Society
 */

#ifndef __PERFORMANCEMEASURES_H_
#define __PERFORMANCEMEASURES_H_

#include "base/SGObject.h"
#include "features/Labels.h"
#include "lib/DynamicArray.h"

/**
 * class to implement various performance measures, like:
 * ROC
 *
 * based on:
 * Fawcett, T: March 2004, ROC Graphs: Notes and Practical
 * Considerations for Researchers
 *
 * @author Sebastian Henschel
 */
class CPerformanceMeasures : public CSGObject
{
	public:
		/** default constructor */
		CPerformanceMeasures();

		/** constructor
		 *
		 * @param true_labels_ true labels as seen in real world
		 * @param output_ output labels/hypothesis from a classifier
		 */
		CPerformanceMeasures(CLabels* true_labels_, CLabels* output_);

		virtual ~CPerformanceMeasures();

		/** initialise performance measures
		 *
		 * @param true_labels_ true labels as seen in real world
		 * @param output_ output labels/hypothesis from a classifier
		 * @return if initialising was successful
		 */
		void init(CLabels* true_labels_, CLabels* output_);

		/** set true labels as seen in real world
		 *
		 * @param true_labels_ true labels
		 * @return if setting was successful
		 */
		inline bool set_true_labels(CLabels* true_labels_)
		{
			true_labels=true_labels_;
			SG_REF(true_labels);
			return true;
		}

		/** get true labels as seen in real world
		 *
		 * @return true labels as seen in real world
		 */
		inline CLabels* get_true_labels() const { return true_labels; }

		/** set output labels/hypothesis from a classifier
		 *
		 * @param output_ output labels
		 * @return if setting was successful
		 */
		inline bool set_output(CLabels* output_)
		{
			output=output_;
			SG_REF(output);
			return true;
		}

		/** get output labels/hypothesis from a classifier
		 *
		 * @return output labels
		 */
		inline CLabels* get_output() const { return output; }

		/** get ROC for labels previously given (swig compatible)
		 * also computes auROC and accROC
		 * caller has to free
		 *
		 * @param result where computed ROC values will be stored
		 * @param dim number of labels/examples
		 * @param num number of elements in each result (== 2)
		 * @return if computation was successful
		 */
		void get_ROC(DREAL** result, INT* dim, INT* num);

		/** return area under ROC
		 *
		 * @return area under ROC
		 */
		inline DREAL get_auROC()
		{
			if (auROC==0) compute_ROC();
			return auROC;
		}


		/** return area over ROC
		 *
		 * @return area over ROC
		 */
		inline DREAL get_aoROC()
		{
			if (auROC==0) compute_ROC();
			return 1.-auROC;
		}

		/** get classifier's accuracy aligned to ROC (swig compatible)
		 * caller has to free
		 *
		 * @param result where accuracy will be stored
		 * @param num number of accuracy values
		 */
		void get_accROC(DREAL** result, INT* num);

		/** get classifier's error rate aligned to ROC (swig compatible)
		 *
		 * @return error rate of classifier
		 */
		void get_errROC(DREAL** result, INT* num);

	protected:
		/** true labels/examples as seen in real world */
		CLabels* true_labels;
		/** output labels/hypothesis from a classifier */
		CLabels* output;
		/** number of true labels/outputs/accuracies/ROC points */
		INT num_labels;

		/** number of positive examples in true_labels */
		INT all_positives;
		/** number of negative examples in true_labels */
		INT all_negatives;

		/** 2 dimensional array of ROC points */
		DREAL* roc;

		/** area under ROC; 1 - area over ROC */
		DREAL auROC;

		/** accuracy of classifier, aligned to ROC; 1 - error */
		DREAL* accROC;

	private:
		/** calculate trapezoid area for auROC
		 *
		 * @param x1 x coordinate of point 1
		 * @param x2 x coordinate of point 2
		 * @param y1 y coordinate of point 1
		 * @param y2 y coordinate of point 2
		 * @return trapezoid area for auROC
		 */
		DREAL trapezoid_area(INT x1, INT x2, INT y1, INT y2);

		/** compute ROC of given labels
		 *
		 * @throws ShogunException
		 */
		void compute_ROC();

};
#endif /* __PERFORMANCEMEASURES_H_ */
