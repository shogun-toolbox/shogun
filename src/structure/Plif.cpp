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

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

#include "structure/Plif.h"

CPlif::CPlif()
{
	limits=NULL ;
	penalties=NULL ;
	cum_derivatives=NULL ;
	id=-1 ;
	next_pen=NULL ;
	transform = T_LINEAR ;
	name = NULL ;
	max_len=0 ;
	min_len=0 ;
	cache=NULL ;
	use_svm=0 ;
	use_cache=false ;
}

CPlif::~CPlif()
{
	//fprintf(stderr, "plif destructor\n") ;
	//if (id!=-1)
	{
		delete[] limits ;
		delete[] penalties ;
		delete[] name ;
		delete[] cache ;
		delete[] cum_derivatives ; 
	}
}

bool CPlif::set_transform_type(const char *type_str) 
{
	delete[] cache ;
	cache=NULL ;

	if (strcmp(type_str, "linear")==0)
		transform = T_LINEAR ;
	else if (strcmp(type_str, "")==0)
		transform = T_LINEAR ;
	else if (strcmp(type_str, "log")==0)
		transform = T_LOG ;
	else if (strcmp(type_str, "log(+1)")==0)
		transform = T_LOG_PLUS1 ;
	else if (strcmp(type_str, "log(+3)")==0)
		transform = T_LOG_PLUS3 ;
	else if (strcmp(type_str, "(+3)")==0)
		transform = T_LINEAR_PLUS3 ;
	else
	{
		CIO::message(M_ERROR, "unknown transform type (%s)\n", type_str) ;
		return false ;
	}
	return true ;
}

void CPlif::init_penalty_struct_cache()
{
	if (!use_cache)
		return ;
	if (cache || use_svm)
		return ;
	if (max_len<0)
		return ;
	//fprintf(stderr, "init cache of size %i byte\n", (max_len+1)*sizeof(DREAL)) ;
	
	cache=new DREAL[max_len+1] ;
	if (cache)
	{
		DREAL input_value ;
		for (INT i=0; i<=max_len; i++)
			if (i<min_len)
				cache[i] = -CMath::INFTY ;
			else
				cache[i] = lookup_penalty(i, 0, false, input_value) ;
	}
}

	
void CPlif::set_name(char *p_name) 
{
	delete[] name ;
	name=new char[strlen(p_name)+1] ;
	strcpy(name,p_name) ;
}

#ifdef HAVE_MATLAB
CPlif* read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT P)
{
	//P = mxGetN(mx_penalty_info) ;
	//fprintf(stderr, "p=%i size=%i\n", P, P*sizeof(CPlif)) ;
	
	CPlif* PEN = new CPlif[P] ;
	
	for (INT i=0; i<P; i++)
	{
		const mxArray* mx_elem = mxGetCell(mx_penalty_info, i) ;
		if (mx_elem==NULL || !mxIsStruct(mx_elem))
		{
			CIO::message(M_ERROR, "empty cell element\n") ;
			delete[] PEN ;
			return NULL ;
		}
		const mxArray* mx_id_field = mxGetField(mx_elem, 0, "id") ;
		if (mx_id_field==NULL || !mxIsNumeric(mx_id_field) || 
			mxGetN(mx_id_field)!=1 || mxGetM(mx_id_field)!=1)
		{
			CIO::message(M_ERROR, "missing id field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_limits_field = mxGetField(mx_elem, 0, "limits") ;
		if (mx_limits_field==NULL || !mxIsNumeric(mx_limits_field) ||
			mxGetM(mx_limits_field)!=1)
		{
			CIO::message(M_ERROR, "missing limits field\n") ;
			delete[] PEN ;
			return NULL ;
		}
		INT len = mxGetN(mx_limits_field) ;
		
		const mxArray* mx_penalties_field = mxGetField(mx_elem, 0, "penalties") ;
		if (mx_penalties_field==NULL || !mxIsNumeric(mx_penalties_field) ||
			mxGetM(mx_penalties_field)!=1 || mxGetN(mx_penalties_field)!=len)
		{
			CIO::message(M_ERROR, "missing penalties field\n") ;
			delete[] PEN ;
			return NULL ;
		}
		const mxArray* mx_transform_field = mxGetField(mx_elem, 0, "transform") ;
		if (mx_transform_field==NULL || !mxIsChar(mx_transform_field))
		{
			CIO::message(M_ERROR, "missing transform field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_name_field = mxGetField(mx_elem, 0, "name") ;
		if (mx_name_field==NULL || !mxIsChar(mx_name_field))
		{
			CIO::message(M_ERROR, "missing name field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_max_len_field = mxGetField(mx_elem, 0, "max_len") ;
		if (mx_max_len_field==NULL || !mxIsNumeric(mx_max_len_field) ||
			mxGetM(mx_max_len_field)!=1 || mxGetN(mx_max_len_field)!=1)
		{
			CIO::message(M_ERROR, "missing max_len field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_min_len_field = mxGetField(mx_elem, 0, "min_len") ;
		if (mx_min_len_field==NULL || !mxIsNumeric(mx_min_len_field) ||
			mxGetM(mx_min_len_field)!=1 || mxGetN(mx_min_len_field)!=1)
		{
			CIO::message(M_ERROR, "missing min_len field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_use_svm_field = mxGetField(mx_elem, 0, "use_svm") ;
		if (mx_use_svm_field==NULL || !mxIsNumeric(mx_use_svm_field) ||
			mxGetM(mx_use_svm_field)!=1 || mxGetN(mx_use_svm_field)!=1)
		{
			CIO::message(M_ERROR, "missing use_svm field\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT use_svm = (INT) mxGetScalar(mx_use_svm_field) ;

		const mxArray* mx_use_cache_field = mxGetField(mx_elem, 0, "use_cache") ;
		if (mx_use_cache_field==NULL || !mxIsNumeric(mx_use_cache_field) ||
			mxGetM(mx_use_cache_field)!=1 || mxGetN(mx_use_cache_field)!=1)
		{
			CIO::message(M_ERROR, "missing use_cache field\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT use_cache = (INT) mxGetScalar(mx_use_cache_field) ;
		
		const mxArray* mx_next_id_field = mxGetField(mx_elem, 0, "next_id") ;
		if (mx_next_id_field==NULL || !mxIsNumeric(mx_next_id_field) ||
			mxGetM(mx_next_id_field)!=1 || mxGetN(mx_next_id_field)!=1)
		{
			CIO::message(M_ERROR, "missing next_id field\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT next_id = (INT) mxGetScalar(mx_next_id_field)-1 ;
		
		INT id = (INT) mxGetScalar(mx_id_field)-1 ;
		if (i<0 || i>P-1)
		{
			CIO::message(M_ERROR, "id out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT max_len = (INT) mxGetScalar(mx_max_len_field) ;
		if (max_len<-1024*1024*100 || max_len>1024*1024*100)
		{
			CIO::message(M_ERROR, "max_len out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id].set_max_len(max_len) ;

		INT min_len = (INT) mxGetScalar(mx_min_len_field) ;
		if (min_len<-1024*1024*100 || min_len>1024*1024*100)
		{
			CIO::message(M_ERROR, "min_len out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id].set_min_len(min_len) ;

		if (PEN[id].get_id()!=-1)
		{
			CIO::message(M_ERROR, "penalty id already used\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id].set_id(id) ;
		if (next_id>=0)
			PEN[id].set_next_pen(&PEN[next_id]) ;
		//fprintf(stderr,"id=%i, next_id=%i\n", id, next_id) ;
		
		ASSERT(next_id!=id) ;
		PEN[id].set_use_svm(use_svm) ;
		PEN[id].set_use_cache(use_cache) ;

		double * limits = mxGetPr(mx_limits_field) ;
		double * penalties = mxGetPr(mx_penalties_field) ;
		PEN[id].set_plif(len, limits, penalties) ;
		
		char *transform_str = mxArrayToString(mx_transform_field) ;				
		char *name_str = mxArrayToString(mx_name_field) ;				

		if (!PEN[id].set_transform_type(transform_str))
		{
			CIO::message(M_ERROR, "transform type not recognized ('%s')\n", transform_str) ;
			delete[] PEN;
			mxFree(transform_str) ;
			return NULL ;
		}
		PEN[id].set_name(name_str) ;
		PEN[id].init_penalty_struct_cache() ;

/*		if (PEN->cache)
/			CIO::message(M_DEBUG, "penalty_info: name=%s id=%i points=%i min_len=%i max_len=%i transform='%s' (cache initialized)\n", PEN[id].name,
					PEN[id].id, PEN[id].len, PEN[id].min_len, PEN[id].max_len, transform_str) ;
		else
			CIO::message(M_DEBUG, "penalty_info: name=%s id=%i points=%i min_len=%i max_len=%i transform='%s'\n", PEN[id].name,
					PEN[id].id, PEN[id].len, PEN[id].min_len, PEN[id].max_len, transform_str) ;
*/
		
		mxFree(transform_str) ;
		mxFree(name_str) ;
	}
	return PEN ;
}
#endif

DREAL CPlif::lookup_penalty_svm(DREAL p_value, DREAL *d_values, bool follow_next, DREAL &input_value) const
{	
	ASSERT(use_svm>0) ;
	DREAL d_value=d_values[use_svm-1] ;
    input_value = d_value ;
	//fprintf(stderr,"transform=%i, d_value=%1.2f\n", (INT)PEN->transform, d_value) ;
	
	switch (transform)
	{
	case T_LINEAR:
		break ;
	case T_LOG:
		d_value = log(d_value) ;
		break ;
	case T_LOG_PLUS1:
		d_value = log(d_value+1) ;
		break ;
	case T_LOG_PLUS3:
		d_value = log(d_value+3) ;
		break ;
	case T_LINEAR_PLUS3:
		d_value = d_value+3 ;
		break ;
	default:
		CIO::message(M_ERROR, "unknown transform\n") ;
		break ;
	}
	
	INT idx = 0 ;
	DREAL ret ;
	for (INT i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
	
	if (idx==0)
		ret=penalties[0] ;
	else if (idx==len)
		ret=penalties[len-1] ;
	else
	{
		ret = (penalties[idx]*(d_value-limits[idx-1]) + penalties[idx-1]*
			   (limits[idx]-d_value)) / (limits[idx]-limits[idx-1]) ;  
	}
	
	if (next_pen && follow_next)
		ret+=next_pen->lookup_penalty(p_value, d_values, follow_next, input_value);
	
	return ret ;
}

DREAL CPlif::lookup_penalty(INT p_value, DREAL* svm_values, bool follow_next, DREAL &input_value) const
{
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values, follow_next, input_value) ;
		
	input_value = (DREAL) p_value ;

	if ((p_value<min_len) || (p_value>max_len))
		return -CMath::INFTY ;
	
	if (cache!=NULL && (p_value>=0) && (p_value<=max_len))
	{
		DREAL ret=cache[p_value] ;
		if (next_pen && follow_next)
			ret+=next_pen->lookup_penalty(p_value, svm_values, true, input_value);
		return ret ;
	}
	return lookup_penalty((DREAL) p_value, svm_values, follow_next, input_value) ;
}

DREAL CPlif::lookup_penalty(DREAL p_value, DREAL* svm_values, bool follow_next, DREAL &input_value) const
{	
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values, follow_next, input_value) ;
		
	input_value = (DREAL) p_value ;

	if ((p_value<min_len) || (p_value>max_len))
		return -CMath::INFTY ;
	
	DREAL d_value = (DREAL) p_value ;
	switch (transform)
	{
	case T_LINEAR:
		break ;
	case T_LOG:
		d_value = log(d_value) ;
		break ;
	case T_LOG_PLUS1:
		d_value = log(d_value+1) ;
		break ;
	case T_LOG_PLUS3:
		d_value = log(d_value+3) ;
		break ;
	case T_LINEAR_PLUS3:
		d_value = d_value+3 ;
		break ;
	default:
		CIO::message(M_ERROR, "unknown transform\n") ;
		break ;
	}

	INT idx = 0 ;
	DREAL ret ;
	for (INT i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
	
	if (idx==0)
		ret=penalties[0] ;
	else if (idx==len)
		ret=penalties[len-1] ;
	else
	{
		ret = (penalties[idx]*(d_value-limits[idx-1]) + penalties[idx-1]*
			   (limits[idx]-d_value)) / (limits[idx]-limits[idx-1]) ;  
	}
	//if (p_value>=30 && p_value<150)
	//fprintf(stderr, "%s %i(%i) -> %1.2f\n", PEN->name, p_value, idx, ret) ;
	
	if (next_pen && follow_next)
		ret+=next_pen->lookup_penalty(p_value, svm_values, true, input_value);

	return ret ;
}

void CPlif::penalty_clear_derivative(bool follow_next) 
{
	for (INT i=0; i<len; i++)
		cum_derivatives[i]=0.0 ;
	
	if (next_pen && follow_next)
		next_pen->penalty_clear_derivative(true);
}

void CPlif::penalty_add_derivative(DREAL p_value, DREAL* svm_values, bool follow_next) 
{
	if (use_svm)
	{
		penalty_add_derivative_svm(p_value, svm_values, follow_next) ;
		return ;
	}
		
	if ((p_value<min_len) || (p_value>max_len))
		return ;
	
	DREAL d_value = (DREAL) p_value ;
	switch (transform)
	{
	case T_LINEAR:
		break ;
	case T_LOG:
		d_value = log(d_value) ;
		break ;
	case T_LOG_PLUS1:
		d_value = log(d_value+1) ;
		break ;
	case T_LOG_PLUS3:
		d_value = log(d_value+3) ;
		break ;
	case T_LINEAR_PLUS3:
		d_value = d_value+3 ;
		break ;
	default:
		CIO::message(M_ERROR, "unknown transform\n") ;
		break ;
	}

	INT idx = 0 ;
	for (INT i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
	
	if (idx==0)
		cum_derivatives[0]+=1 ;
	else if (idx==len)
		cum_derivatives[len-1]+=1 ;
	else
	{
		cum_derivatives[idx]+=(d_value-limits[idx-1])/(limits[idx]-limits[idx-1]) ;
		cum_derivatives[idx-1]+=(limits[idx]-d_value)/(limits[idx]-limits[idx-1]) ;
	}
	
	if (next_pen && follow_next)
		next_pen->penalty_add_derivative(p_value, svm_values, true);

}

void CPlif::penalty_add_derivative_svm(DREAL p_value, DREAL *d_values, bool follow_next) 
{	
	ASSERT(use_svm>0) ;
	DREAL d_value=d_values[use_svm-1] ;
	
	switch (transform)
	{
	case T_LINEAR:
		break ;
	case T_LOG:
		d_value = log(d_value) ;
		break ;
	case T_LOG_PLUS1:
		d_value = log(d_value+1) ;
		break ;
	case T_LOG_PLUS3:
		d_value = log(d_value+3) ;
		break ;
	case T_LINEAR_PLUS3:
		d_value = d_value+3 ;
		break ;
	default:
		CIO::message(M_ERROR, "unknown transform\n") ;
		break ;
	}
	
	INT idx = 0 ;
	for (INT i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
	
	if (idx==0)
		cum_derivatives[0]+=1 ;
	else if (idx==len)
		cum_derivatives[len-1]+=1 ;
	else
	{
		cum_derivatives[idx]+=(d_value-limits[idx-1])/(limits[idx]-limits[idx-1]) ;
		cum_derivatives[idx-1]+=(limits[idx]-d_value)/(limits[idx]-limits[idx-1]) ;
	}

	if (next_pen && follow_next)
		next_pen->penalty_add_derivative(p_value, d_values, follow_next);
}
