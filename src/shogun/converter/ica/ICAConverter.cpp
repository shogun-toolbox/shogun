/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#include <shogun/converter/ica/ICAConverter.h>

#ifdef HAVE_EIGEN3

using namespace shogun;

CICAConverter::CICAConverter() : CConverter()
{	
	init();
}

void CICAConverter::init()
{
	m_mixing_matrix = SGMatrix<float64_t>();
	SG_ADD(&m_mixing_matrix, "mixing_matrix", "m_mixing_matrix", MS_NOT_AVAILABLE);
}

CICAConverter::~CICAConverter()
{
}

SGMatrix<float64_t> CICAConverter::get_mixing_matrix() const
{
	return m_mixing_matrix;
}

#endif // HAVE_EIGEN3
