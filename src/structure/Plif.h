
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_H__
#define __PLIF_H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "structure/PlifBase.h"

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

class CPlif: public CPlifBase
{
public:
	CPlif(INT len=0) ;
	~CPlif() ;
	void init_penalty_struct_cache() ;
	
	DREAL lookup_penalty_svm(DREAL p_value, DREAL *d_values) const;
	DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const ;
	DREAL lookup_penalty(INT p_value, DREAL* svm_values) const ;
	
	inline DREAL lookup(DREAL p_value)
	{
		ASSERT(use_svm == 0);
		return lookup_penalty(p_value, NULL);
	}

	void penalty_clear_derivative() ;
	void penalty_add_derivative_svm(DREAL p_value, DREAL* svm_values) ;
	void penalty_add_derivative(DREAL p_value, DREAL* svm_values) ;
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
	INT get_id() const 
	{
		return id ;
	}
	INT get_max_id() const
		{
			return get_id() ;
		}

	void set_use_svm(INT p_use_svm) 
	{
		delete[] cache ;
		cache=NULL ;
		use_svm=p_use_svm ;
	}
	INT get_use_svm() const 
	{
		return use_svm ;
	}
	virtual bool uses_svm_values() const
		{
			return (get_use_svm()!=0) ;
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

		penalty_clear_derivative() ;
	}

	void set_plif_length(INT p_len) 
	{
		if (len!=p_len)
		{
			len=p_len ;
			delete[] limits ;
			delete[] penalties ;
			delete[] cum_derivatives ;
			SG_DEBUG( "set_plif len=%i\n", p_len);
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
		penalty_clear_derivative() ;
	}

	void set_plif_limits(DREAL* p_limits, INT p_len) 
	{
		delete[] cache ;
		cache=NULL ;
		ASSERT(len==p_len);

		for (INT i=0; i<len; i++)
			limits[i]=p_limits[i] ;

		penalty_clear_derivative() ;
	}

	void set_plif_penalty(DREAL* p_penalties, INT p_len) 
	{
		delete[] cache ;
		cache=NULL ;
		ASSERT(len==p_len);

		for (INT i=0; i<len; i++)
			penalties[i]=p_penalties[i] ;

		penalty_clear_derivative() ;
	}

	inline void set_max_value(DREAL p_max_value) 
	{
		delete[] cache ;
		cache=NULL ;
		max_value=p_max_value ;
	}

	virtual DREAL get_max_value() const 
	{
		return max_value ;
	}

	inline void set_min_value(DREAL p_min_value) 
	{
		delete[] cache ;
		cache=NULL ;
		min_value=p_min_value ;
	}

	virtual DREAL get_min_value() const 
	{
		return min_value ;
	}

	void set_name(char *p_name) ;
	inline char * get_name() 
	{
		return name ;
	}

	inline INT get_plif_len()
	{
		return len ;
	}

protected:

	INT len ;
	DREAL *limits ;
	DREAL *penalties ;
	DREAL *cum_derivatives ;
	DREAL max_value ;
	DREAL min_value ;
	DREAL *cache ;
	enum ETransformType transform ;
	INT id ;
	char * name ;
	INT use_svm ;
	bool use_cache ;
} ;

#ifdef HAVE_MATLAB
CPlif** read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT P) ;
void delete_penalty_struct(CPlif** PEN, INT P) ;
#endif

#endif
