/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */


#include <stdio.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <structure/Plif.h>
#include <lib/memory.h>

//#define PLIF_DEBUG

using namespace shogun;

CPlif::CPlif(int32_t l)
: CPlifBase()
{
	limits=SGVector<float64_t>();
	penalties=SGVector<float64_t>();
	cum_derivatives=SGVector<float64_t>();
	id=-1;
	transform=T_LINEAR;
	name=NULL;
	max_value=0;
	min_value=0;
	cache=NULL;
	use_svm=0;
	use_cache=false;
	len=0;
	do_calc = true;
	if (l>0)
		set_plif_length(l);
}

CPlif::~CPlif()
{
	SG_FREE(name);
	SG_FREE(cache);
}

bool CPlif::set_transform_type(const char *type_str)
{
	invalidate_cache();

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
		SG_ERROR("unknown transform type (%s)\n", type_str)
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

	float64_t* local_cache=SG_MALLOC(float64_t,  ((int32_t) max_value) + 2);

	if (local_cache)
	{
		for (int32_t i=0; i<=max_value; i++)
		{
			if (i<min_value)
				local_cache[i] = -CMath::INFTY ;
			else
				local_cache[i] = lookup_penalty(i, NULL) ;
		}
	}
	this->cache=local_cache ;
}

void CPlif::set_plif_name(char *p_name)
{
	SG_FREE(name);
	name=get_strdup(p_name);
}

char* CPlif::get_plif_name() const
{
	if (name)
		return name;
	else
	{
		char buf[20];
		sprintf(buf, "plif%i", id);
		return get_strdup(buf);
	}
}

void CPlif::delete_penalty_struct(CPlif** PEN, int32_t P)
{
	for (int32_t i=0; i<P; i++)
		delete PEN[i] ;
	SG_FREE(PEN);
}

float64_t CPlif::lookup_penalty_svm(
	float64_t p_value, float64_t *d_values) const
{
	ASSERT(use_svm>0)
	float64_t d_value=d_values[use_svm-1] ;
#ifdef PLIF_DEBUG
	SG_PRINT("%s.lookup_penalty_svm(%f)\n", get_name(), d_value)
#endif

	if (!do_calc)
		return d_value;
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
		SG_ERROR("unknown transform\n")
		break ;
	}

	int32_t idx = 0 ;
	float64_t ret ;
	for (int32_t i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
		else
			break ; // assume it is monotonically increasing

#ifdef PLIF_DEBUG
	SG_PRINT("  -> idx = %i ", idx)
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
		SG_PRINT("  -> (%1.3f*%1.3f, %1.3f*%1.3f)", (d_value-limits[idx-1])/(limits[idx]-limits[idx-1]), penalties[idx], (limits[idx]-d_value)/(limits[idx]-limits[idx-1]), penalties[idx-1])
#endif
	}
#ifdef PLIF_DEBUG
		SG_PRINT("  -> ret=%1.3f\n", ret)
#endif

	return ret ;
}

float64_t CPlif::lookup_penalty(int32_t p_value, float64_t* svm_values) const
{
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values) ;

	if ((p_value<min_value) || (p_value>max_value))
	{
		//SG_PRINT("Feature:%s, %s.lookup_penalty(%i): return -inf min_value: %f, max_value: %f\n", name, get_name(), p_value, min_value, max_value)
		return -CMath::INFTY ;
	}
	if (!do_calc)
		return p_value;
	if (cache!=NULL && (p_value>=0) && (p_value<=max_value))
	{
		float64_t ret=cache[p_value] ;
		return ret ;
	}
	return lookup_penalty((float64_t) p_value, svm_values) ;
}

float64_t CPlif::lookup_penalty(float64_t p_value, float64_t* svm_values) const
{
	if (use_svm)
		return lookup_penalty_svm(p_value, svm_values) ;

#ifdef PLIF_DEBUG
	SG_PRINT("%s.lookup_penalty(%f)\n", get_name(), p_value)
#endif


	if ((p_value<min_value) || (p_value>max_value))
	{
		//SG_PRINT("Feature:%s, %s.lookup_penalty(%f): return -inf min_value: %f, max_value: %f\n", name, get_name(), p_value, min_value, max_value)
		return -CMath::INFTY ;
	}

	if (!do_calc)
		return p_value;

	float64_t d_value = (float64_t) p_value ;
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
		SG_ERROR("unknown transform\n")
		break ;
	}

#ifdef PLIF_DEBUG
	SG_PRINT("  -> value = %1.4f ", d_value)
#endif

	int32_t idx = 0 ;
	float64_t ret ;
	for (int32_t i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
		else
			break ; // assume it is monotonically increasing

#ifdef PLIF_DEBUG
	SG_PRINT("  -> idx = %i ", idx)
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
		SG_PRINT("  -> (%1.3f*%1.3f, %1.3f*%1.3f) ", (d_value-limits[idx-1])/(limits[idx]-limits[idx-1]), penalties[idx], (limits[idx]-d_value)/(limits[idx]-limits[idx-1]), penalties[idx-1])
#endif
	}
	//if (p_value>=30 && p_value<150)
	//SG_PRINT("%s %i(%i) -> %1.2f\n", PEN->name, p_value, idx, ret)
#ifdef PLIF_DEBUG
	SG_PRINT("  -> ret=%1.3f\n", ret)
#endif

	return ret ;
}

void CPlif::penalty_clear_derivative()
{
	for (int32_t i=0; i<len; i++)
		cum_derivatives[i]=0.0 ;
}

void CPlif::penalty_add_derivative(float64_t p_value, float64_t* svm_values, float64_t factor)
{
	if (use_svm)
	{
		penalty_add_derivative_svm(p_value, svm_values, factor) ;
		return ;
	}

	if ((p_value<min_value) || (p_value>max_value))
	{
		return ;
	}
	float64_t d_value = (float64_t) p_value ;
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
		SG_ERROR("unknown transform\n")
		break ;
	}

	int32_t idx = 0 ;
	for (int32_t i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
		else
			break ; // assume it is monotonically increasing

	if (idx==0)
		cum_derivatives[0]+= factor ;
	else if (idx==len)
		cum_derivatives[len-1]+= factor ;
	else
	{
		cum_derivatives[idx] += factor * (d_value-limits[idx-1])/(limits[idx]-limits[idx-1]) ;
		cum_derivatives[idx-1]+= factor*(limits[idx]-d_value)/(limits[idx]-limits[idx-1]) ;
	}
}

void CPlif::penalty_add_derivative_svm(float64_t p_value, float64_t *d_values, float64_t factor)
{
	ASSERT(use_svm>0)
	float64_t d_value=d_values[use_svm-1] ;

	if (d_value<-1e+20)
		return;

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
		SG_ERROR("unknown transform\n")
		break ;
	}

	int32_t idx = 0 ;
	for (int32_t i=0; i<len; i++)
		if (limits[i]<=d_value)
			idx++ ;
		else
			break ; // assume it is monotonically increasing

	if (idx==0)
		cum_derivatives[0]+=factor ;
	else if (idx==len)
		cum_derivatives[len-1]+=factor ;
	else
	{
		cum_derivatives[idx] += factor*(d_value-limits[idx-1])/(limits[idx]-limits[idx-1]) ;
		cum_derivatives[idx-1] += factor*(limits[idx]-d_value)/(limits[idx]-limits[idx-1]) ;
	}
}

void CPlif::get_used_svms(int32_t* num_svms, int32_t* svm_ids)
{
	if (use_svm)
	{
		svm_ids[(*num_svms)] = use_svm;
		(*num_svms)++;
	}
	SG_PRINT("->use_svm:%i plif_id:%i name:%s trans_type:%s  ",use_svm, get_id(), get_name(), get_transform_type())
}

bool CPlif::get_do_calc()
{
	return do_calc;
}

void CPlif::set_do_calc(bool b)
{
	do_calc = b;;
}
