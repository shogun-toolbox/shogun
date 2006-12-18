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
}

CPlifArray::~CPlifArray()
{
}

DREAL CPlifArray::lookup_penalty(DREAL p_value, DREAL* svm_values) const 
{
	DREAL ret = 0.0 ;
	for (INT i=0; i<m_array.get_array_size(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
} 

DREAL CPlifArray::lookup_penalty(INT p_value, DREAL* svm_values) const 
{
	DREAL ret = 0.0 ;
	for (INT i=0; i<m_array.get_array_size(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
} 

void CPlifArray::penalty_clear_derivative() 
{
	for (INT i=0; i<m_array.get_array_size(); i++)
		m_array[i]->penalty_clear_derivative() ;
} 

void CPlifArray::penalty_add_derivative(DREAL p_value, DREAL* svm_values)
{
	for (INT i=0; i<m_array.get_array_size(); i++)
		m_array[i]->penalty_add_derivative(p_value, svm_values) ;
}

DREAL CPlifArray::get_max_value() const 
{
	DREAL min_max_value = 0 ;
	for (INT i=0; i<m_array.get_array_size(); i++)
		min_max_value = CMath::min(min_max_value, m_array[i]->get_max_value()) ;
	return min_max_value ;
}

DREAL CPlifArray::get_min_value() const 
{
	DREAL max_min_value = 0 ;
	for (INT i=0; i<m_array.get_array_size(); i++)
		max_min_value = CMath::max(max_min_value, m_array[i]->get_min_value()) ;
	return max_min_value ;
}

bool CPlifArray::uses_svm_values() const 
{
	for (INT i=0; i<m_array.get_array_size(); i++)
		if (m_array[i]->uses_svm_values())
			return true ;
	return false ;
}

INT CPlifArray::get_max_id() const 
{
	INT max_id = 0 ;
	for (INT i=0; i<m_array.get_array_size(); i++)
		max_id = CMath::max(max_id, m_array[i]->get_max_id()) ;
	return max_id ;
}



