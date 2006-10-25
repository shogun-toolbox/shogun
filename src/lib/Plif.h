/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_H__
#define __PLIF_H__

#include "lib/common.h"
#include "lib/Mathematics.h"

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

enum ETransformType
{
	T_LINEAR,
	T_LOG,
	T_LOG_PLUS1,
	T_LOG_PLUS3,
	T_LINEAR_PLUS3
}  ;

#ifdef HAVE_MATLAB
	void read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT &P) ;
#endif

class CPlif
{
public:
	CPlif() ;
	~CPlif() ;
	//void delete_penalty_struct_array(struct penalty_struct *PEN, INT len) ;
	void init_penalty_struct_cache() ;
	
	
	DREAL lookup_penalty_svm(INT p_value, DREAL *d_values, bool follow_next, DREAL &input_value) ;
	DREAL lookup_penalty(INT p_value, DREAL* svm_values, bool follow_next, DREAL &input_value) ;
	
	bool set_transform_type(const char *type_str) ;
	
	void set_id(INT p_id) 
	{
		id=p_id ;
	}
	INT get_id() 
	{
		return id ;
	}
	void set_next_pen(CPlif * p_next_pen) 
	{
		next_pen=p_next_pen ;
	}

	void set_use_svm(INT p_use_svm) 
	{
		use_svm=p_use_svm ;
	}
	void set_plif(INT p_len, DREAL *p_limits, DREAL* p_penalties) 
	{
		len=p_len ;
		delete[] limits ;
		delete[] penalties ;
		limits=new DREAL[len] ;
		penalties=new DREAL[len] ;
		for (INT i=0; i<len; i++)
		{
			limits[i]=p_limits[i] ;
			penalties[i]=p_penalties[i] ;
		}
	}
	void set_max_len(INT p_max_len) 
	{
		max_len=p_max_len ;
	}
	void set_min_len(INT p_min_len) 
	{
		min_len=p_min_len ;
	}
	void set_name(char *p_name) ;
	
protected:

	INT len ;
	DREAL *limits ;
	DREAL *penalties ;
	INT max_len ;
	INT min_len ;
	DREAL *cache ;
	enum ETransformType transform ;
	INT id ;
	CPlif *next_pen ;
	char * name ;
	INT use_svm ;
} ;


#endif
