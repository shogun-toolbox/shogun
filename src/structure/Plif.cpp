/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include <stdio.h>
#include <string.h>

#include "lib/io.h"

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

#include "structure/Plif.h"

#define PLIF_DEBUG

CPlif::CPlif(INT l)
{
	limits=NULL ;
	penalties=NULL ;
	cum_derivatives=NULL ;
	id=-1 ;
	transform = T_LINEAR ;
	name = NULL ;
	max_value=0 ;
	min_value=0 ;
	cache=NULL ;
	use_svm=0 ;
	use_cache=false ;
	len=0;

	if (l>0)
		set_plif_length(l);
}

CPlif::~CPlif()
{
	delete[] limits ;
	delete[] penalties ;
	delete[] name ;
	delete[] cache ;
	delete[] cum_derivatives ; 
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
		SG_ERROR( "unknown transform type (%s)\n", type_str) ;
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
	if (max_value<=0)
		return ;

	DREAL* local_cache=new DREAL[ ((INT) max_value) + 2] ;
	
	if (local_cache)
	{
		for (INT i=0; i<=max_value; i++)
		{
			if (i<min_value)
				local_cache[i] = -CMath::INFTY ;
			else
				local_cache[i] = lookup_penalty(i, NULL) ;
		}
	}
	this->cache=local_cache ;
}

	
void CPlif::set_name(char *p_name) 
{
	delete[] name ;
	name=new char[strlen(p_name)+1] ;
	strcpy(name,p_name) ;
}

#ifdef HAVE_MATLAB
CPlif** read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT P)
{
	//P = mxGetN(mx_penalty_info) ;
	//fprintf(stderr, "p=%i size=%i\n", P, P*sizeof(CPlif)) ;
	
	CPlif** PEN = new CPlif*[P] ;
	for (INT i=0; i<P; i++)
		PEN[i]=new CPlif() ;
	
	for (INT i=0; i<P; i++)
	{
		//fprintf(stderr, "i=%i/%i\n", i, P) ;
		
		const mxArray* mx_elem = mxGetCell(mx_penalty_info, i) ;
		if (mx_elem==NULL || !mxIsStruct(mx_elem))
		{
			SG_SERROR("empty cell element\n") ;
			delete[] PEN ;
			return NULL ;
		}
		const mxArray* mx_id_field = mxGetField(mx_elem, 0, "id") ;
		if (mx_id_field==NULL || !mxIsNumeric(mx_id_field) || 
			mxGetN(mx_id_field)!=1 || mxGetM(mx_id_field)!=1)
		{
			SG_SERROR( "missing id field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_limits_field = mxGetField(mx_elem, 0, "limits") ;
		if (mx_limits_field==NULL || !mxIsNumeric(mx_limits_field) ||
			mxGetM(mx_limits_field)!=1)
		{
			SG_SERROR( "missing limits field\n") ;
			delete[] PEN ;
			return NULL ;
		}
		INT len = mxGetN(mx_limits_field) ;
		
		const mxArray* mx_penalties_field = mxGetField(mx_elem, 0, "penalties") ;
		if (mx_penalties_field==NULL || !mxIsNumeric(mx_penalties_field) ||
			mxGetM(mx_penalties_field)!=1 || mxGetN(mx_penalties_field)!=len)
		{
			SG_SERROR( "missing penalties field (%i)\n", i) ;
			delete[] PEN ;
			return NULL ;
		}
		const mxArray* mx_transform_field = mxGetField(mx_elem, 0, "transform") ;
		if (mx_transform_field==NULL || !mxIsChar(mx_transform_field))
		{
			SG_SERROR( "missing transform field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_name_field = mxGetField(mx_elem, 0, "name") ;
		if (mx_name_field==NULL || !mxIsChar(mx_name_field))
		{
			SG_SERROR( "missing name field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_max_value_field = mxGetField(mx_elem, 0, "max_value") ;
		if (mx_max_value_field==NULL || !mxIsNumeric(mx_max_value_field) ||
			mxGetM(mx_max_value_field)!=1 || mxGetN(mx_max_value_field)!=1)
		{
			SG_SERROR( "missing max_value field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_min_value_field = mxGetField(mx_elem, 0, "min_value") ;
		if (mx_min_value_field==NULL || !mxIsNumeric(mx_min_value_field) ||
			mxGetM(mx_min_value_field)!=1 || mxGetN(mx_min_value_field)!=1)
		{
			SG_SERROR( "missing min_value field\n") ;
			delete[] PEN;
			return NULL ;
		}
		const mxArray* mx_use_svm_field = mxGetField(mx_elem, 0, "use_svm") ;
		if (mx_use_svm_field==NULL || !mxIsNumeric(mx_use_svm_field) ||
			mxGetM(mx_use_svm_field)!=1 || mxGetN(mx_use_svm_field)!=1)
		{
			SG_SERROR( "missing use_svm field\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT use_svm = (INT) mxGetScalar(mx_use_svm_field) ;

		const mxArray* mx_use_cache_field = mxGetField(mx_elem, 0, "use_cache") ;
		if (mx_use_cache_field==NULL || !mxIsNumeric(mx_use_cache_field) ||
			mxGetM(mx_use_cache_field)!=1 || mxGetN(mx_use_cache_field)!=1)
		{
			SG_SERROR( "missing use_cache field\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT use_cache = (INT) mxGetScalar(mx_use_cache_field) ;

		INT id = (INT) mxGetScalar(mx_id_field)-1 ;
		if (i<0 || i>P-1)
		{
			SG_SERROR( "id out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		INT max_value = (INT) mxGetScalar(mx_max_value_field) ;
		if (max_value<-1024*1024*100 || max_value>1024*1024*100)
		{
			SG_SERROR( "max_value out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id]->set_max_value(max_value) ;

		INT min_value = (INT) mxGetScalar(mx_min_value_field) ;
		if (min_value<-1024*1024*100 || min_value>1024*1024*100)
		{
			SG_SERROR( "min_value out of range\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id]->set_min_value(min_value) ;
		
		if (PEN[id]->get_id()!=-1)
		{
			SG_SERROR( "penalty id already used\n") ;
			delete[] PEN;
			return NULL ;
		}
		PEN[id]->set_id(id) ;
		
		PEN[id]->set_use_svm(use_svm) ;
		PEN[id]->set_use_cache(use_cache) ;

		double * limits = mxGetPr(mx_limits_field) ;
		double * penalties = mxGetPr(mx_penalties_field) ;
		PEN[id]->set_plif(len, limits, penalties) ;
		
		char *transform_str = mxArrayToString(mx_transform_field) ;				
		char *name_str = mxArrayToString(mx_name_field) ;				

		if (!PEN[id]->set_transform_type(transform_str))
		{
			SG_SERROR( "transform type not recognized ('%s')\n", transform_str) ;
			delete[] PEN;
			mxFree(transform_str) ;
			return NULL ;
		}

		PEN[id]->set_name(name_str) ;
		PEN[id]->init_penalty_struct_cache() ;

/*		if (PEN->cache)
/			SG_SDEBUG( "penalty_info: name=%s id=%i points=%i min_value=%i max_value=%i transform='%s' (cache initialized)\n", PEN[id]->name,
					PEN[id]->id, PEN[id]->len, PEN[id]->min_value, PEN[id]->max_value, transform_str) ;
		else
			SG_SDEBUG( "penalty_info: name=%s id=%i points=%i min_value=%i max_value=%i transform='%s'\n", PEN[id]->name,
					PEN[id]->id, PEN[id]->len, PEN[id]->min_value, PEN[id]->max_value, transform_str) ;
*/
		
		mxFree(transform_str) ;
		mxFree(name_str) ;
	}
	return PEN ;
}

void delete_penalty_struct(CPlif** PEN, INT P) 
{
	for (INT i=0; i<P; i++)
		delete PEN[i] ;
	delete[] PEN ;
}

#endif

DREAL CPlif::lookup_penalty_svm(DREAL p_value, DREAL *d_values) const
{	
	ASSERT(use_svm>0) ;
	DREAL d_value=d_values[use_svm-1] ;
#ifdef PLIF_DEBUG
	SG_DEBUG(stderr, "lookup_penalty_svm(%f)\n", d_value) ;
#endif

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
		SG_ERROR("unknown transform\n");
		break ;
	}
	
	INT idx = 0 ;
	DREAL ret ;
	for (INT i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
#ifdef PLIF_DEBUG
	SG_DEBUG(stderr, "  -> idx = %i ", idx) ;
#endif
	
	if (idx==0)
		ret=penalties[0] ;
	else if (idx==len)
		ret=penalties[len-1] ;
	else
	{
		ret = (penalties[idx]*(d_value-limits[idx-1]) + penalties[idx-1]*
			   (limits[idx]-d_value)) / (limits[idx]-limits[idx-1]) ;  
#ifdef PLIF_DEBUG
		SG_DEBUG(stderr, "  -> (%1.3f,%1.3f)", (d_value-limits[idx-1])/(limits[idx]-limits[idx-1]), (limits[idx]-d_value)/(limits[idx]-limits[idx-1])) ;
#endif
	}
#ifdef PLIF_DEBUG
		SG_DEBUG(stderr, "  -> ret=%1.3f", ret) ;
#endif
	
	return ret ;
}

DREAL CPlif::lookup_penalty(INT p_value, DREAL* svm_values) const
{
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values) ;

	if ((p_value<min_value) || (p_value>max_value))
		return -CMath::INFTY ;
	
	if (cache!=NULL && (p_value>=0) && (p_value<=max_value))
	{
		DREAL ret=cache[p_value] ;
		return ret ;
	}
	return lookup_penalty((DREAL) p_value, svm_values) ;
}

DREAL CPlif::lookup_penalty(DREAL p_value, DREAL* svm_values) const
{	
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values) ;

	if ((p_value<min_value) || (p_value>max_value))
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
		SG_ERROR( "unknown transform\n") ;
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
	
	return ret ;
}

void CPlif::penalty_clear_derivative() 
{
	for (INT i=0; i<len; i++)
		cum_derivatives[i]=0.0 ;
}

void CPlif::penalty_add_derivative(DREAL p_value, DREAL* svm_values) 
{
	if (use_svm)
	{
		penalty_add_derivative_svm(p_value, svm_values) ;
		return ;
	}
	
	if ((p_value<min_value) || (p_value>max_value))
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
		SG_ERROR( "unknown transform\n") ;
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
}

void CPlif::penalty_add_derivative_svm(DREAL p_value, DREAL *d_values) 
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
		SG_ERROR( "unknown transform\n") ;
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
}
