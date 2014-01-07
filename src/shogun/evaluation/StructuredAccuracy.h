/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef __STRUCTURED_ACCURACY_H__
#define __STRUCTURED_ACCURACY_H__

#include <evaluation/Evaluation.h>
#include <labels/StructuredLabels.h>

namespace shogun
{

/**
 * @brief class CStructuredAccuracy used to compute accuracy of structured classification
 */
class CStructuredAccuracy : public CEvaluation
{
	public:
		/** default constructor */
		CStructuredAccuracy();

		/** destructor */
		virtual ~CStructuredAccuracy();

		/** evaluate accuracy
		 *
		 * @param predicted labels to be evaluated
		 * @param ground_truth labels assumed to be correct
		 *
		 * @return accuracy
		 */
		virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

		/** NOT IMPLEMENTED
		 * constructs confusion matrix for multiclass classification
		 *
		 * @param predicted labels to be evaluated
		 * @param ground_truth labels assumed to be correct
		 *
		 * @return confusion matrix
		 */
		static SGMatrix<int32_t> get_confusion_matrix(CLabels* predicted, CLabels* ground_truth);

		/** whether the evaluation criterion has to be maximimed or minimized
		*
		* @return maximize evaluation criterion
		*/
		inline EEvaluationDirection get_evaluation_direction() const
		{
			return ED_MAXIMIZE;
		}

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "StructuredAccuracy"; }

	private:
		/** evaluate accuracy for structured labels composed of real numbers
		 *
		 * @param predicted labels to be evaluated
		 * @param ground_truth labels assumed to be correct
		 *
		 * @return accuracy
		 */
		float64_t evaluate_real(CStructuredLabels* predicted, CStructuredLabels* ground_truth);

		/** evaluate accuracy for structured labels composed of sequences
		 *
		 * @param predicted labels to be evaluated
		 * @param ground_truth labels assumed to be correct
		 *
		 * @return accuracy
		 */
		float64_t evaluate_sequence(CStructuredLabels* predicted, CStructuredLabels* ground_truth);

}; /* class CStructuredAccuracy*/

} /* namespace shogun */

#endif /* __STRUCTURED_ACCURACY_H__ */
