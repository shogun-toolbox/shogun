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

		/** compute ROC for selected pair of labels
		 *
		 * @param result where computed ROC values will be stored
		 * @param dim number of labels/examples
		 * @param num number of elements in each result (== 2)
		 * @throws ShogunException
		 * @return if computation was successful
		 */
		void compute_ROC(DREAL** result, INT* dim, INT* num);

	protected:
		/** true labels/examples as seen in real world */
		CLabels* true_labels;
		/** number of positive examples in true_labels */
		INT all_positives;
		/** number of negative examples in true_labels */
		INT all_negatives;
		/** sorted true labels as seen in real world */
		DREAL* sorted_true_labels;

		/** output labels/hypothesis from a classifier */
		CLabels* output;
		/** sorted output labels/hypothesis from a classifier */
		DREAL* sorted_output;

	private:
		/** sort for ROC
		 * results are stored in descending order in sorted_*
		 */
		void ROC_sort();
};
#endif /* __PERFORMANCEMEASURES_H_ */
