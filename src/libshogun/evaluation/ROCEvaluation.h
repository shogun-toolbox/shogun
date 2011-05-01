/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ROCEVALUATION_H_
#define ROCEVALUATION_H_

#include "evaluation/BinaryClassEvaluation.h"

namespace shogun
{

class CLabels;

/** @brief The class ROCEvalution
 * used to evaluate ROC (Receiver Operator Characteristic)
 * graph of binary classifier. This class also has an capability
 * of calculating auROC (area under ROC).
 *
 * Implementation is based on the efficient ROC algorithm described in
 *
 * Fawcett, Tom (2004) ROC Graphs:
 * Notes and Practical Considerations for Researchers; Machine Learning, 2004
 */
class CROCEvaluation: public CBinaryClassEvaluation
{
public:
	/** constructor */
	CROCEvaluation() :
		CBinaryClassEvaluation(), m_ROC_graph(NULL),
		m_auROC(0.0), m_ROC_length(0), m_computed(false) {};

	/** destructor */
	virtual ~CROCEvaluation();

	/** get name */
	virtual inline const char* get_name() const { return "ROCEvaluation"; };

	/** evaluate ROC and auROC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auROC
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** get auROC
	 * @return area under ROC (auROC)
	 */
	float64_t get_auROC();

	/** get ROC (swig)
	 * @param result matrix of ROC graph
	 * @param num number of points in ROC graph
	 * @param dim dimensionality (always 2)
	 */
	void get_ROC(float64_t** result, int32_t* num, int32_t* dim);

protected:

	/** 2-d array used to store ROC graph */
	float64_t* m_ROC_graph;

	/** area under ROC graph */
	float64_t m_auROC;

	/** number of points in ROC graph */
	int32_t m_ROC_length;

	/** indicator of ROC and auROC being computed already */
	bool m_computed;
};

}

#endif /* ROCEVALUATION_H_ */
