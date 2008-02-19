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

/** class to implement various performance measures:
 * Receiver Operating Curve (ROC),
 * Precision Recall Curve (PRC),
 * Detection Error Tradeoff (DET),
 * Cross Correlation coefficient (CC),
 * Weighted Relative Accuracy (WRAcc),
 * Balanced Error (BAL)
 *
 * based on:
 * Fawcett, T: March 2004, ROC Graphs: Notes and Practical
 * Considerations for Researchers
 * Sonnenburg, S: Feburary 2008, various discussions
 */
class CPerformanceMeasures : public CSGObject
{
	public:
		/** default constructor */
		CPerformanceMeasures();

		/** constructor
		 *
		 * @param true_labels true labels as seen in real world
		 * @param output output labels/hypothesis from a classifier
		 */
		CPerformanceMeasures(CLabels* true_labels, CLabels* output);

		virtual ~CPerformanceMeasures();

		/** initialise performance measures
		 *
		 * @param true_labels true labels as seen in real world
		 * @param output output labels/hypothesis from a classifier
		 */
		void init(CLabels* true_labels, CLabels* output);

		/** set true labels as seen in real world
		 *
		 * @param true_labels true labels
		 * @return if setting was successful
		 */
		inline bool set_true_labels(CLabels* true_labels)
		{
			m_true_labels=true_labels;
			SG_REF(true_labels);
			return true;
		}

		/** get true labels as seen in real world
		 *
		 * @return true labels as seen in real world
		 */
		inline CLabels* get_true_labels() const { return m_true_labels; }

		/** set output labels/hypothesis from a classifier
		 *
		 * @param output output labels
		 * @return if setting was successful
		 */
		inline bool set_output(CLabels* output)
		{
			m_output=output;
			SG_REF(output);
			return true;
		}

		/** get classifier's output labels/hypothesis
		 *
		 * @return output labels
		 */
		inline CLabels* get_output() const { return m_output; }

		/** get number of labels in output/true labels
		 *
		 * @return number of labels in output/true labels
		 */
		inline INT get_num_labels() const { return m_num_labels; }

		/** get Receiver Operating Curve for previously given labels
		 * (swig compatible)
		 *
		 * ROC point = false positives / all false labels,
		 *             true positives / all true labels
		 *
		 * caller has to free
		 *
		 * @param result where computed ROC values will be stored
		 * @param dim number of labels/examples
		 * @param num number of elements in each result (== 2)
		 */
		void get_ROC(DREAL** result, INT* dim, INT* num);

		/** return area under Receiver Operating Curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under ROC
		 */
		inline DREAL get_auROC()
		{
			if (m_auROC==CMath::ALMOST_NEG_INFTY) {
				DREAL** roc=(DREAL**) malloc(sizeof(DREAL**));
				compute_ROC(roc);
				free(*roc);
				free(roc);
			}
			return m_auROC;
		}

		/** return area over Reveiver Operating Curve
		 *
		 * value is 1 - auROC
		 *
		 * @return area over ROC
		 */
		inline DREAL get_aoROC()
		{
			return 1.0-get_auROC();
		}

		/** get classifier's accuracies aligned to Receiver Operating Curve
		 * (swig compatible)
		 *
		 * accuracy = (true positives + true negatives) / all labels
		 *
		 * caller has to free
		 *
		 * @param result where accuracies will be stored
		 * @param num number of accuracy values
		 */
		void get_accuracyROC(DREAL** result, INT* num);

		/** get classifier's accuracy at threshold
		 *
		 * @param threshold all values below are considered negative examples 
		 *        (default 0)
		 * @return classifer's accuracy at threshold
		 */
		DREAL get_accuracy(DREAL threshold=0);

		/** get classifier's error rates aligned to Receiver Operating Curve
		 * (swig compatible)
		 *
		 * value is 1 - accuracy
		 *
		 * caller has to free
		 *
		 * @param result where errors will be stored
		 * @param num number of error values
		 */
		void get_errorROC(DREAL** result, INT* num);

		/** get classifier's error at threshold
		 *
		 * value is 1 - accuracy0
		 *
		 * @param threshold all values below are considered negative examples
		 *        (default 0)
		 * @return classifer's error at threshold
		 */
		inline DREAL get_error(DREAL threshold=0)
		{
			return 1.0-get_accuracy(threshold);
		}

		/** get Precision Recall Curve for previously given labels
		 * (swig compatible)
		 *
		 * PRC point = true positives / all true labels,
		 *             true positives / (true positives + false positives)
		 *
		 * caller has to free
		 *
		 * @param result where computed ROC values will be stored
		 * @param dim number of labels/examples
		 * @param num number of elements in each result (== 2)
		 */
		void get_PRC(DREAL** result, INT* dim, INT* num);

		/** return area under Precision Recall Curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under PRC
		 */
		inline DREAL get_auPRC()
		{
			if (m_auPRC==CMath::ALMOST_NEG_INFTY) {
				DREAL** prc=(DREAL**) malloc(sizeof(DREAL**));
				compute_PRC(prc);
				free(*prc);
				free(prc);
			}
			return m_auPRC;
		}

		/** return area over Precision Recall Curve
		 *
		 * value is 1 - auPRC
		 *
		 * @return area over PRC
		 */
		inline DREAL get_aoPRC()
		{
			return 1-get_auPRC();
		}

		/** get classifier's F-measure aligned to Precision Recall Curve
		 * (swig compatible)
		 *
		 * F-measure = 2 / (1 / precision + 1 / recall)
		 *
		 * caller has to free
		 *
		 * @param result where F-measure will be stored
		 * @param num number of F-measure values
		 */
		void get_fmeasurePRC(DREAL** result, INT* num);

		/** get classifier's F-measure at threshold 0
		 *
		 * @return classifer's F-measure at threshold 0
		 */
		DREAL get_fmeasure(DREAL threshold=0);

		/** get Detection Error Tradeoff curve for previously given labels
		 * (swig compatible)
		 *
		 * DET point = false positives / all false labels,
		 *             false negatives / all false labels
		 *
		 * caller has to free
		 *
		 * @param result where computed DET values will be stored
		 * @param dim number of labels/examples
		 * @param num number of elements in each result (== 2)
		 */
		void get_DET(DREAL** result, INT* dim, INT* num);

		/** return area under Detection Error Tradeoff curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under DET curve
		 */
		inline DREAL get_auDET()
		{
			if (m_auDET==CMath::ALMOST_NEG_INFTY) {
				DREAL** det=(DREAL**) malloc(sizeof(DREAL**));
				compute_DET(det);
				free(*det);
				free(det);
			}
			return m_auDET;
		}

		/** return area over Detection Error Tradeoff curve
		 *
		 * value is 1 - auDET
		 *
		 * @return area over DET curve
		 */
		inline DREAL get_aoDET()
		{
			return 1-get_auDET();
		}

		/** get classifier's Cross Correlation coefficients (swig compatible)
		 *
		 * CC = (
		 *        true positives * true negatives
		 *        -
		 *        false positives * false negatives
		 *      )
		 *      /
		 *      sqrt(
		 *        (true positives + false positives)
		 *        *
		 *        (true positives + false negatives)
		 *        *
		 *        (true negatives + false positives)
		 *        *
		 *        (true negatives + false negatives)
		 *      )
		 *
		 * also check http://en.wikipedia.org/wiki/Correlation
		 *
		 * caller has to free
		 *
		 * @param result where CCs will be stored
		 * @param num number of CC values
		 */
		void get_all_CC(DREAL** result, INT* num);

		/** get classifier's Cross Correlation coefficient at threshold
		 *
		 * @return classifer's CC at threshold 
		 */
		DREAL get_CC(DREAL threshold=0);

		/** get classifier's Weighted Relative Accuracy (swig compatible)
		 *
		 * WRAcc = (
		 *           true positives / (true positives + false negatives)
		 *         )
		 *         -
		 *         (
		 *           false positives / (false positives + true negatives)
		 *         )
		 *
		 * caller has to free
		 *
		 * @param result where WRAcc values will be stored
		 * @param num number of WRAcc values
		 */
		void get_all_WRAcc(DREAL** result, INT* num);

		/** get classifier's Weighted Relative Accuracy at threshold 0
		 *
		 * @return classifer's WRAcc at threshold 0
		 */
		DREAL get_WRAcc(DREAL threshold=0);

		/** get classifier's Balanced Error (swig compatible)
		 *
		 * BAL = 0.5
		 *       *
		 *       (
		 *         true positives / all true labels
		 *         +
		 *         true negatives / all false labels
		 *       )
		 *
		 * caller has to free
		 *
		 * @param result where BAL values will be stored
		 * @param num number of BAL values
		 */
		void get_all_BAL(DREAL** result, INT* num);

		/** get classifier's Balanced Error at threshold 0
		 *
		 * @return classifer's BAL at threshold 0
		 */
		DREAL get_BAL(DREAL threshold=0);

	protected:
		/** calculate trapezoid area for auROC
		 *
		 * @param x1 x coordinate of point 1
		 * @param x2 x coordinate of point 2
		 * @param y1 y coordinate of point 1
		 * @param y2 y coordinate of point 2
		 * @return trapezoid area for auROC
		 */
		template <class T> DREAL trapezoid_area(T x1, T x2, T y1, T y2);

		/** create index for ROC sorting
		 *
		 */
		void create_sortedROC();
		
		/** compute ROC points and auROC
		 *
		 */
		void compute_ROC(DREAL** result);

		/** compute ROC accuracy/error
		 *
		 * @param result where the result will be stored
		 * @param do_error if error instead of accuracy shall be computed
		 */
		void compute_accuracyROC(DREAL** result, bool do_error=false);

		/** compute PRC points and auPRC
		 *
		 * @param result where the result will be stored
		 */
		void compute_PRC(DREAL** result);

		/** compute DET points and auDET
		 *
		 * @param result where the result will be stored
		 */
		void compute_DET(DREAL** result);

		/** compute confusion matrix 
		 *
		 * caller has to delete[]
		 *
		 * @param threshold threshold to check against
		 * @return array of 4 values, the number of:
		 *         true positives, false positives, false negatives, true
		 *         negatives
		 */
		INT* compute_confusion_matrix(DREAL threshold);

	protected:
		/** true labels/examples as seen in real world */
		CLabels* m_true_labels;
		/** output labels/hypothesis from a classifier */
		CLabels* m_output;
		/** number of true labels/outputs/accuracies/ROC points */
		INT m_num_labels;

		/** number of positive examples in true_labels */
		INT m_all_true;
		/** number of negative examples in true_labels */
		INT m_all_false;

		/** array of size num_labels with indices of true_labels/output
		 * sorted to fit ROC algorithm */
		INT* m_sortedROC;
		/** area under ROC; 1 - area over ROC */
		DREAL m_auROC;
		/** classifier's accuracy */
		DREAL m_accuracy;

		/** area under PRC; 1 - area over PRC */
		DREAL m_auPRC;
		/** classifier's F-measure */
		DREAL m_fmeasure;

		/** area under DET; 1 - area over DET */
		DREAL m_auDET;

		/** classifier's CC */
		DREAL m_cc;
		/** classifier's weighted relative accuracy */
		DREAL m_wracc;
		/** classifier's balanced error */
		DREAL m_bal;
};
#endif /* __PERFORMANCEMEASURES_H_ */
