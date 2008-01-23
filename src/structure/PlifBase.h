/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_BASE_H__
#define __PLIF_BASE_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Mathematics.h"

class CPlifBase : public CSGObject
{
public:
	CPlifBase() {} ;
	virtual ~CPlifBase() {} ;
	
	virtual DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const =0 ;
	virtual DREAL lookup_penalty(INT p_value, DREAL* svm_values) const =0 ;
	
	virtual void penalty_clear_derivative()=0 ;
	virtual void penalty_add_derivative(DREAL p_value, DREAL* svm_values)=0 ;
	
	virtual DREAL get_max_value() const =0 ;
	virtual DREAL get_min_value() const =0 ;

	virtual bool uses_svm_values() const = 0 ;
	virtual INT get_max_id() const = 0 ;
};
#endif
