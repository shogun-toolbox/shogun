/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "BinaryClassEvaluation.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

using namespace shogun;

void CBinaryClassEvaluation::get_scores(CLabels* predicted, CLabels* ground_truth)
{
	m_TP = 0.0;
	m_FP = 0.0;
	m_TN = 0.0;
	m_FN = 0.0;
	for(int i=0; i<predicted->get_num_labels(); i++)
	{
		if (CMath::sign(ground_truth->get_label(i))==1)
		{
			if (CMath::sign(predicted->get_label(i))==1)
				m_TP += 1.0;
			else
				m_FN += 1.0;
		}
		else
		{
			if (CMath::sign(predicted->get_label(i))==1)
				m_FP += 1.0;
			else
				m_TN += 1.0;
		}
	}
}
