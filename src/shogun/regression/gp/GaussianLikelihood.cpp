/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/GaussianLikelihood.h>

namespace shogun {

CGaussianLikelihood::CGaussianLikelihood() {
	m_sigma = 0.01;

}

CGaussianLikelihood::~CGaussianLikelihood() {
	// TODO Auto-generated destructor stub
}


SGVector<float64_t> CGaussianLikelihood::evaluate_means(SGVector<float64_t>& means)
{
	return SGVector<float64_t>(means);
}

SGVector<float64_t> CGaussianLikelihood::evaluate_variances(SGVector<float64_t>& vars)
{
	SGVector<float64_t> result(vars);
	for(int i = 0; i < result.vlen; i++) result[i] += (m_sigma*m_sigma);
	return result;
}


}
