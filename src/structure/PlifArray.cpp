/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include <stdio.h>
#include <string.h>

#include "lib/io.h"

#include "structure/PlifArray.h"

CPlifArray::CPlifArray()
	: m_array(5)
{
	min_value = -1e6 ;
	max_value = 1e6 ;
}

CPlifArray::~CPlifArray()
{
	m_array.zero() ;
}

void CPlifArray::add_plif(CPlifBase* new_plif) 
{
	ASSERT(new_plif!=NULL) ;
	m_array.append_element(new_plif) ;
	//fprintf(stderr, "m_array.get_num_elements()=%i\n", m_array.get_num_elements()) ;
	
	min_value = -1e6 ;
	for (INT i=0; i<m_array.get_num_elements(); i++)
	{
		ASSERT(m_array[i]!=NULL)
		if (!m_array[i]->uses_svm_values())
			min_value = CMath::max(min_value, m_array[i]->get_min_value()) ;
	}
	
	max_value = 1e6 ;
	for (INT i=0; i<m_array.get_num_elements(); i++)
		if (!m_array[i]->uses_svm_values())
			max_value = CMath::min(max_value, m_array[i]->get_max_value()) ;
}

void CPlifArray::clear() 
{
	m_array.resize_array(0) ;
	min_value = -1e6 ;
	max_value = 1e6 ;
	//fprintf(stderr, "clear: m_array.get_num_elements()=%i\n", m_array.get_num_elements()) ;
}

DREAL CPlifArray::lookup_penalty(DREAL p_value, DREAL* svm_values) const 
{
	if (p_value<min_value || p_value>max_value)
		return -CMath::INFTY ;

	DREAL ret = 0.0 ;
	for (INT i=0; i<m_array.get_num_elements(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
} 

DREAL CPlifArray::lookup_penalty(INT p_value, DREAL* svm_values) const 
{
	if (p_value<min_value || p_value>max_value)
		return -CMath::INFTY ;
	
	DREAL ret = 0.0 ;
	for (INT i=0; i<m_array.get_num_elements(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
} 

void CPlifArray::penalty_clear_derivative() 
{
	for (INT i=0; i<m_array.get_num_elements(); i++)
		m_array[i]->penalty_clear_derivative() ;
} 

void CPlifArray::penalty_add_derivative(DREAL p_value, DREAL* svm_values)
{
	for (INT i=0; i<m_array.get_num_elements(); i++)
		m_array[i]->penalty_add_derivative(p_value, svm_values) ;
}

bool CPlifArray::uses_svm_values() const 
{
	for (INT i=0; i<m_array.get_num_elements(); i++)
		if (m_array[i]->uses_svm_values())
			return true ;
	return false ;
}

INT CPlifArray::get_max_id() const 
{
	INT max_id = 0 ;
	for (INT i=0; i<m_array.get_num_elements(); i++)
		max_id = CMath::max(max_id, m_array[i]->get_max_id()) ;
	return max_id ;
}



