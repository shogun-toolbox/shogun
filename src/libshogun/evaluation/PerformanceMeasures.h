/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Sebastian Henschel
 * Copyright (C) 2008-2009 Friedrich Miescher Laboratory of Max-Planck-Society
 */

#ifndef __PERFORMANCEMEASURES_H_
#define __PERFORMANCEMEASURES_H_

#include "base/SGObject.h"
#include "features/Labels.h"
#include "lib/DynamicArray.h"

namespace shogun
{
    class CLabels;

/** @brief Class to implement various performance measures.
 *
 * - Receiver Operating Curve (ROC)
 * - Area under the ROC curve (auROC)
 * - Area over the ROC curve (aoROC)
 * - Precision Recall Curve (PRC)
 * - Area under the PRC (auPRC)
 * - Area over the PRC (aoPRC)
 * - Detection Error Tradeoff (DET)
 * - Area under the DET (auDET)
 * - Area over the DET (aoDET)
 * - Cross Correlation coefficient (CC)
 * - Weighted Relative Accuracy (WRAcc)
 * - Balanced Error (BAL)
 * - F-Measure
 * - Accuracy
 * - Error
 *
 * based on:
 * Fawcett, T: March 2004, ROC Graphs: Notes and Practical
 * Considerations for Researchers and input from
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

		/** set true labels as seen in real world
		 *
		 * @param true_labels true labels
		 * @return if setting was successful
		 */
		bool set_true_labels(CLabels* true_labels);

		/** get true labels as seen in real world
		 *
		 * @return true labels as seen in real world
		 */
		inline CLabels* get_true_labels() const
		{
			SG_REF(m_true_labels);
			return m_true_labels;
		}

		/** set output labels/hypothesis from a classifier
		 *
		 * @param output output labels
		 * @return if setting was successful
		 */
		bool set_output(CLabels* output);

		/** get classifier's output labels/hypothesis
		 *
		 * @return output labels
		 */
		inline CLabels* get_output() const
		{
			SG_REF(m_output);
			return m_output;
		}

		/** get number of labels in output/true labels
		 *
		 * @return number of labels in output/true labels
		 */
		inline int32_t get_num_labels() const { return m_num_labels; }

		/** get Receiver Operating Curve for previously given labels
		 * (swig compatible)
		 *
		 * ROC point = false positives / all false labels,
		 *             true positives / all true labels
		 *
		 * caller has to free
		 *
		 * @param result where computed ROC values will be stored
		 * @param num number of labels/examples
		 * @param dim dimensions == 2 (false positive rate, true positive rate)
		 */
		void get_ROC(float64_t** result, int32_t* num, int32_t* dim);

		/** return area under Receiver Operating Curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under ROC
		 */
		inline float64_t get_auROC()
		{
			if (m_auROC==CMath::ALMOST_NEG_INFTY)
			{
				float64_t** roc=(float64_t**) malloc(sizeof(float64_t**));
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
		inline float64_t get_aoROC()
		{
			return 1.0-get_auROC();
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
		 * @param num number of labels/examples
		 * @param dim dimension == 2 (recall, precision)
		 */
		void get_PRC(float64_t** result, int32_t* num, int32_t* dim);

		/** return area under Precision Recall Curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under PRC
		 */
		inline float64_t get_auPRC()
		{
			if (m_auPRC==CMath::ALMOST_NEG_INFTY) {
				float64_t** prc=(float64_t**) malloc(sizeof(float64_t**));
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
		inline float64_t get_aoPRC()
		{
			return 1-get_auPRC();
		}

		/** get Detection Error Tradeoff curve for previously given labels
		 * (swig compatible)
		 *
		 * DET point = false positives / all false labels,
		 *             false negatives / all false labels
		 *
		 * caller has to free
		 *
		 * @param result where computed DET values will be stored
		 * @param num number of labels/examples
		 * @param dim dimension == 2 (false positive rate, false negative rate)
		 */
		void get_DET(float64_t** result, int32_t* num, int32_t* dim);

		/** return area under Detection Error Tradeoff curve
		 *
		 * calculated by adding trapezoids
		 *
		 * @return area under DET curve
		 */
		inline float64_t get_auDET()
		{
			if (m_auDET==CMath::ALMOST_NEG_INFTY)
			{
				float64_t** det=(float64_t**) malloc(sizeof(float64_t**));
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
		inline float64_t get_aoDET()
		{
			return 1-get_auDET();
		}

		/** get classifier's accuracies (swig compatible)
		 *
		 * accuracy = (true positives + true negatives) / all labels
		 *
		 * caller has to free
		 *
		 * @param result storage of accuracies in 2 dim array: (output, accuracy),
		 *               sorted by output
		 * @param num number of accuracy points
		 * @param dim dimension == 2 (output, accuracy)
		 */
		void get_all_accuracy(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's accuracy at given threshold
		 *
		 * @param threshold all values below are considered negative examples
		 *        (default 0)
		 * @return classifer's accuracy at threshold
		 */
		float64_t get_accuracy(float64_t threshold=0);

		/** get classifier's error rates (swig compatible)
		 *
		 * value is 1 - accuracy
		 *
		 * caller has to free
		 *
		 * @param result storage of errors in 2 dim array: (output, error),
		 *               sorted by output
		 * @param num number of accuracy points
		 * @param dim dimension == 2 (output, error)
		 */
		void get_all_error(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's error at threshold
		 *
		 * value is 1 - accuracy0
		 *
		 * @param threshold all values below are considered negative examples
		 *        (default 0)
		 * @return classifer's error at threshold
		 */
		inline float64_t get_error(float64_t threshold=0)
		{
			return 1.0-get_accuracy(threshold);
		}

		/** get classifier's F-measure (swig compatible)
		 *
		 * F-measure = 2 / (1 / precision + 1 / recall)
		 *
		 * caller has to free
		 *
		 * @param result storage of F-measure in 2 dim array (output, fmeasure),
		 *               sorted by output
		 * @param num number of accuracy points
		 * @param dim dimension == 2 (output, fmeasure)
		 */
		void get_all_fmeasure(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's F-measure at threshold 0
		 *
		 * @return classifer's F-measure at threshold 0
		 */
		float64_t get_fmeasure(float64_t threshold=0);

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
		 * @param result storage of CCs in 2 dim array: (output, CC), sorted
		 *               by output
		 * @param num number of CC points
		 * @param dim dimension == 2 (output, CC)
		 */
		void get_all_CC(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's Cross Correlation coefficient at threshold
		 *
		 * @return classifer's CC at threshold 
		 */
		float64_t get_CC(float64_t threshold=0);

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
		 * @param result storage of WRAccs in 2 dim array: (output, WRAcc),
		 *               sorted by output
		 * @param num number of WRAcc points
		 * @param dim dimension == 2 (output, WRAcc)
		 */
		void get_all_WRAcc(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's Weighted Relative Accuracy at threshold 0
		 *
		 * @return classifer's WRAcc at threshold 0
		 */
		float64_t get_WRAcc(float64_t threshold=0);

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
		 * @param result storage of BAL in 2 dim array: (output, BAL),
		 *               sorted by output
		 * @param num number of BAL points
		 * @param dim dimension == 2 (output, BAL)
		 */
		void get_all_BAL(float64_t** result, int32_t* num, int32_t* dim);

		/** get classifier's Balanced Error at threshold 0
		 *
		 * @return classifer's BAL at threshold 0
		 */
		float64_t get_BAL(float64_t threshold=0);

        /** get the name of tghe object
         *
         * @return name of object
         */
        inline virtual const char* get_name() const { return "PerformanceMeasures"; }

	protected:
		/** initialise values independent from true labels/output */
		void init_nolabels();

		/** calculate trapezoid area for auROC
		 *
		 * @param x1 x coordinate of point 1
		 * @param x2 x coordinate of point 2
		 * @param y1 y coordinate of point 1
		 * @param y2 y coordinate of point 2
		 * @return trapezoid area for auROC
		 */
		float64_t trapezoid_area(float64_t x1, float64_t x2, float64_t y1, float64_t y2);

		/** create index for ROC sorting
		 *
		 */
		void create_sortedROC();
		
		/** compute ROC points and auROC
		 *
		 */
		void compute_ROC(float64_t** result);

		/** compute ROC accuracy/error
		 *
		 * @param result where the result will be stored
		 * @param num number of points
		 * @param dim dimension == 2 (output, accuracy/error)
		 * @param do_error if error instead of accuracy shall be computed
		 */
		void compute_accuracy(
			float64_t** result, int32_t* num, int32_t* dim, bool do_error=false);

		/** compute PRC points and auPRC
		 *
		 * @param result where the result will be stored
		 */
		void compute_PRC(float64_t** result);

		/** compute DET points and auDET
		 *
		 * @param result where the result will be stored
		 */
		void compute_DET(float64_t** result);

		/** compute confusion matrix
		 *
		 * caller has to delete[]
		 *
		 * @param threshold threshold to compute against
		 * @param tp storage of true positives or NULL if unused
		 * @param fp storage of false positives or NULL if unused
		 * @param fn storage of false negatives or NULL if unused
		 * @param tn storage of true negatives or NULL if unused
		 */
		void compute_confusion_matrix(
			float64_t threshold,
			int32_t* tp, int32_t* fp, int32_t* fn, int32_t* tn);

	protected:
		/** true labels/examples as seen in real world */
		CLabels* m_true_labels;
		/** output labels/hypothesis from a classifier */
		CLabels* m_output;
		/** number of true labels/outputs/accuracies/ROC points */
		int32_t m_num_labels;

		/** number of positive examples in true_labels */
		int32_t m_all_true;
		/** number of negative examples in true_labels */
		int32_t m_all_false;

		/** array of size num_labels with indices of true_labels/output
		 * sorted to fit ROC algorithm */
		int32_t* m_sortedROC;
		/** area under ROC; 1 - area over ROC */
		float64_t m_auROC;
		/** area under PRC; 1 - area over PRC */
		float64_t m_auPRC;
		/** area under DET; 1 - area over DET */
		float64_t m_auDET;
};
} // namespace shogun
#endif /* __PERFORMANCEMEASURES_H_ */
