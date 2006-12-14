
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

class CPlif
{
public:
	CPlif(INT len=0) ;
	~CPlif() ;
	void init_penalty_struct_cache() ;
	
	DREAL lookup_penalty_svm(DREAL p_value, DREAL *d_values, bool follow_next, DREAL &input_value) const ;
	DREAL lookup_penalty(DREAL p_value, DREAL* svm_values, bool follow_next, DREAL &input_value) const ;
	DREAL lookup_penalty(INT p_value, DREAL* svm_values, bool follow_next, DREAL &input_value) const ;

	inline DREAL lookup_penalty(DREAL p_value)
	{
		DREAL dummy;
		ASSERT(use_svm == 0);
		return lookup_penalty(p_value, NULL, false, dummy);
	}

	inline DREAL lookup_penalty(INT p_value)
	{
		DREAL dummy;
		ASSERT(use_svm == 0);
		return lookup_penalty(p_value, NULL, false, dummy);
	}

	void penalty_clear_derivative(bool follow_next) ;
	void penalty_add_derivative_svm(DREAL p_value, DREAL* svm_values, bool follow_next) ;
	void penalty_add_derivative(DREAL p_value, DREAL* svm_values, bool follow_next) ;
	const DREAL * get_cum_derivative(INT & p_len) const 
	{
		p_len = len ;
		return cum_derivatives ;
	}
	
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
	CPlif* get_next_pen() 
	{
		return next_pen ;
	}

	void set_use_svm(INT p_use_svm) 
	{
		delete[] cache ;
		cache=NULL ;
		use_svm=p_use_svm ;
	}
	INT get_use_svm()
	{
		return use_svm ;
	}

	void set_use_cache(INT p_use_cache) 
	{
		delete[] cache ;
		cache=NULL ;
		use_cache=p_use_cache ;
	}
	INT get_use_cache()
	{
		return use_cache ;
	}

	// for swig use set_plif_len, set_plif_limits, set_plif_penalty
	void set_plif(INT p_len, DREAL *p_limits, DREAL* p_penalties) 
	{
		len=p_len ;
		delete[] limits ;
		delete[] penalties ;
		delete[] cum_derivatives ;
		delete[] cache ;
		cache=NULL ;

		limits=new DREAL[len] ;
		penalties=new DREAL[len] ;
		cum_derivatives=new DREAL[len] ;

		for (INT i=0; i<len; i++)
		{
			limits[i]=p_limits[i] ;
			penalties[i]=p_penalties[i] ;
		}

		penalty_clear_derivative(false) ;
	}

	void set_plif_length(INT p_len) 
	{
		if (len!=p_len)
		{
			len=p_len ;
			delete[] limits ;
			delete[] penalties ;
			delete[] cum_derivatives ;
			CIO::message(M_DEBUG, "set_plif len=%i\n", p_len);
			limits=new DREAL[len] ;
			penalties=new DREAL[len] ;
			cum_derivatives=new DREAL[len] ;
		}
		delete[] cache ;
		cache=NULL ;
		for (INT i=0; i<len; i++)
		{
			limits[i]=0.0 ;
			penalties[i]=0.0 ;
		}
		penalty_clear_derivative(false) ;
	}

	void set_plif_limits(DREAL* p_limits, INT p_len) 
	{
		delete[] cache ;
		cache=NULL ;
		ASSERT(len==p_len);

		for (INT i=0; i<len; i++)
			limits[i]=p_limits[i] ;

		penalty_clear_derivative(false) ;
	}

	void set_plif_penalty(DREAL* p_penalties, INT p_len) 
	{
		delete[] cache ;
		cache=NULL ;
		ASSERT(len==p_len);

		for (INT i=0; i<len; i++)
			penalties[i]=p_penalties[i] ;

		penalty_clear_derivative(false) ;
	}

	void set_max_len(INT p_max_len) 
	{
		delete[] cache ;
		cache=NULL ;
		max_len=p_max_len ;
	}
	INT get_max_len() const 
	{
		return max_len ;
	}
	void set_min_len(INT p_min_len) 
	{
		delete[] cache ;
		cache=NULL ;
		min_len=p_min_len ;
	}
	void set_name(char *p_name) ;
	char * get_name() 
	{
		return name ;
	}

	INT get_plif_len()
	{
		return len ;
	}

	DREAL get_plif_loss()
	{
		return loss ;
	}
	void set_plif_loss(DREAL p_loss)
	{
		loss = p_loss ;
	}
	
	
protected:

	INT len ;
	DREAL *limits ;
	DREAL *penalties ;
	DREAL *cum_derivatives ;
	INT max_len ;
	INT min_len ;
	DREAL *cache ;
	enum ETransformType transform ;
	INT id ;
	CPlif *next_pen ;
	char * name ;
	INT use_svm ;
	bool use_cache ;
	DREAL loss ;
} ;

#ifdef HAVE_MATLAB
CPlif* read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT P) ;
#endif

#endif
