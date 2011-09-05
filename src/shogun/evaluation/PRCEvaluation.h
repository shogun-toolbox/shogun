/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef PRCEVALUATION_H_
#define PRCEVALUATION_H_

#include <shogun/evaluation/BinaryClassEvaluation.h>

namespace shogun
{

class CLabels;

/** @brief Class PRCEvaluation used to evaluate PRC 
 * (Precision Recall Curve) and an area under PRC curve (auPRC).
 *
 */
class CPRCEvaluation: public CBinaryClassEvaluation
{
public:
	/** constructor */
	CPRCEvaluation() :
		CBinaryClassEvaluation(), m_PRC_graph(NULL),
		m_auPRC(0.0), m_PRC_length(0), m_computed(false) {};

	/** destructor */
	virtual ~CPRCEvaluation();

	/** get name */
	virtual inline const char* get_name() const { return "PRCEvaluation"; };

	/** evaluate PRC and auPRC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auPRC
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	inline EEvaluationDirection get_evaluation_direction()
	{
		return ED_MAXIMIZE;
	}

	/** get auPRC
	 * @return area under PRC (auPRC)
	 */
	float64_t get_auPRC();

	/** get PRC
	 * @return PRC graph matrix
	 */
	SGMatrix<float64_t> get_PRC();

protected:

	/** 2-d array used to store PRC graph */
	float64_t* m_PRC_graph;

	/** area under PRC graph */
	float64_t m_auPRC;

	/** number of points in PRC graph */
	int32_t m_PRC_length;

	/** indicator of PRC and auPRC being computed already */
	bool m_computed;
};

}

#endif /* PRCEVALUATION_H_ */
