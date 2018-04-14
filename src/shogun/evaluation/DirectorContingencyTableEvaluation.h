/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Bjoern Esser, Viktor Gal
 */

#ifndef DIRECTORCONTINGENCYTABLEEVALUATION_H_
#define DIRECTORCONTINGENCYTABLEEVALUATION_H_

#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/lib/config.h>
#ifdef USE_SWIG_DIRECTORS
namespace shogun
{

#define IGNORE_IN_CLASSLIST

/** @brief The class DirectorContingencyTableEvaluation
 * a base class used to evaluate 2-class classification
 * using SWIG directors.
 */
IGNORE_IN_CLASSLIST class CDirectorContingencyTableEvaluation: public CContingencyTableEvaluation
{

public:

	/** constructor */
	CDirectorContingencyTableEvaluation() :
		CContingencyTableEvaluation(CUSTOM)
	{
	}

	/** destructor */
	virtual ~CDirectorContingencyTableEvaluation()
	{
	}

	/** Evaluate */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth)
	{
		return CContingencyTableEvaluation::evaluate(predicted, ground_truth);
	}

	/** Computes custom score, not implemented
	 * @return custom score value
	 */
	virtual float64_t get_custom_score()
	{
		SG_NOTIMPLEMENTED
		return 0.0;
	}

	/** Returns custom direction, not implemented
	 * @return direction of custom score
	 */
	virtual EEvaluationDirection get_custom_direction()
	{
		SG_NOTIMPLEMENTED
		return ED_MAXIMIZE;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "DirectorContingencyTableEvaluation";
	}

};

}
#endif /* USE_SWIG_DIRECTORS */
#endif /* DIRECTORCONTINGENCYTABLEEVALUATION_H_ */
