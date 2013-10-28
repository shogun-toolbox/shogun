/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#include <stdio.h>
#include <string.h>

#include <shogun/io/SGIO.h>

#include <shogun/structure/PlifArray.h>
#include <shogun/structure/Plif.h>

//#define PLIFARRAY_DEBUG

using namespace shogun;

CPlifArray::CPlifArray()
: CPlifBase()
{
	min_value=-1e6;
	max_value=1e6;
}

CPlifArray::~CPlifArray()
{
}

void CPlifArray::add_plif(CPlifBase* new_plif)
{
	ASSERT(new_plif)
	m_array.append_element(new_plif) ;

	min_value = -1e6 ;
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
	{
		ASSERT(m_array[i])
		if (!m_array[i]->uses_svm_values())
			min_value = CMath::max(min_value, m_array[i]->get_min_value()) ;
	}

	max_value = 1e6 ;
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		if (!m_array[i]->uses_svm_values())
			max_value = CMath::min(max_value, m_array[i]->get_max_value()) ;
}

void CPlifArray::clear()
{
	m_array.clear_array(NULL);
	min_value = -1e6 ;
	max_value = 1e6 ;
}

float64_t CPlifArray::lookup_penalty(
	float64_t p_value, float64_t* svm_values) const
{
	//min_value = -1e6 ;
	//max_value = 1e6 ;
	if (p_value<min_value || p_value>max_value)
	{
		//SG_WARNING("lookup_penalty: p_value: %i min_value: %f, max_value: %f\n",p_value, min_value, max_value)
		return -CMath::INFTY ;
	}
	float64_t ret = 0.0 ;
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
}

float64_t CPlifArray::lookup_penalty(
	int32_t p_value, float64_t* svm_values) const
{
	//min_value = -1e6 ;
	//max_value = 1e6 ;
	if (p_value<min_value || p_value>max_value)
	{
		//SG_WARNING("lookup_penalty: p_value: %i min_value: %f, max_value: %f\n",p_value, min_value, max_value)
		return -CMath::INFTY ;
	}
	float64_t ret = 0.0 ;
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
	{
		float64_t val = m_array[i]->lookup_penalty(p_value, svm_values) ;
		ret += val ;
#ifdef PLIFARRAY_DEBUG
		CPlif * plif = (CPlif*)m_array[i] ;
		if (plif->get_use_svm())
			SG_PRINT("penalty[%i]=%1.5f (use_svm=%i -> %1.5f)\n", i, val, plif->get_use_svm(), svm_values[plif->get_use_svm()-1])
		else
			SG_PRINT("penalty[%i]=%1.5f\n", i, val)
#endif
	}
	return ret ;
}

void CPlifArray::penalty_clear_derivative()
{
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		m_array[i]->penalty_clear_derivative() ;
}

void CPlifArray::penalty_add_derivative(
	float64_t p_value, float64_t* svm_values, float64_t factor)
{
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		m_array[i]->penalty_add_derivative(p_value, svm_values, factor) ;
}

bool CPlifArray::uses_svm_values() const
{
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		if (m_array[i]->uses_svm_values())
			return true ;
	return false ;
}

int32_t CPlifArray::get_max_id() const
{
	int32_t max_id = 0 ;
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
		max_id = CMath::max(max_id, m_array[i]->get_max_id()) ;
	return max_id ;
}

void CPlifArray::get_used_svms(int32_t* num_svms, int32_t* svm_ids)
{
	SG_PRINT("get_used_svms: num: %i \n",m_array.get_num_elements())
	for (int32_t i=0; i<m_array.get_num_elements(); i++)
	{
		m_array[i]->get_used_svms(num_svms, svm_ids);
	}
	SG_PRINT("\n")
}
