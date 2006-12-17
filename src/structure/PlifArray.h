
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIFARRAY_H__
#define __PLIFARRAY_H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/Array.h"
#include "structure/PlifBase.h"

class CPlifArray: public CPlifBase
{
public:
	CPlifArray()  ;
	virtual ~CPlifArray()  ;
	
	void add_plif(CPlifBase* new_plif) 
		{
			INT prev_num = m_array.get_array_size() ;
			m_array.resize_array(prev_num+1) ;
			m_array[prev_num] = new_plif ;
		}
	void clear() 
		{
			m_array.resize_array(0) ;
		}
	INT get_num_plifs()
		{
			return m_array.get_array_size() ;
		}
	

	virtual DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const  ;
	virtual DREAL lookup_penalty(INT p_value, DREAL* svm_values) const  ;
	
	virtual void penalty_clear_derivative() ;
	virtual void penalty_add_derivative(DREAL p_value, DREAL* svm_values) ;
	
	virtual DREAL get_max_value() const ;
	virtual DREAL get_min_value() const ;

	virtual bool uses_svm_values() const ;
	virtual INT get_max_id() const ;
protected:
	CArray<CPlifBase*> m_array ;
} ;

#endif
