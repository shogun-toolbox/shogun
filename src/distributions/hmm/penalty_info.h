#ifndef __PENALTY_INFO_H__
#define __PENALTY_INFO_H__

#include "lib/common.h"
#include "lib/Mathmatics.h"
#include <mex.h>

enum ETransformType
{
	T_LINEAR,
	T_LOG,
	T_LOG_PLUS3,
	T_LINEAR_PLUS3
}  ;

struct penalty_struct
{
	INT len ;
	REAL *limits ;
	REAL *penalties ;
	INT max_len ;
	REAL *cache ;
	enum ETransformType transform ;
	INT id ;
	char * name ;
} ;

void init_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct_array(struct penalty_struct *PEN, INT len) ;

#ifdef MATLAB
struct penalty_struct * read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT &P) ;
#endif

inline REAL lookup_step_penalty(INT id, INT len)
{
	switch (id)
	{
	case 0: 
		return 0 ;
	case 1:
        if (len%3==0) return 0 ; else return -math.INFTY ;
	case 2:
        if (len%3==1) return 0 ; else return -math.INFTY ;
	case 3:
        if (len%3==2) return 0 ; else return -math.INFTY ;
	case 10:
        if (len>1) return 0 ; else return -math.INFTY ;
	case 11:
        if ((len%3==0) && (len>1)) return 0 ; else return -math.INFTY ;
	case 12:
        if ((len%3==1) && (len>1)) return 0 ; else return -math.INFTY ;
	case 13:
        if ((len%3==2) && (len>1)) return 0 ; else return -math.INFTY ;
	case 5:
        if (len==2) return 0 ; else return -math.INFTY ;
	}
	CIO::message(M_ERROR, "unknown step-penalty\n") ;
	return -math.INFTY ;
}

inline REAL lookup_penalty(const struct penalty_struct *PEN, INT value)
{
	//fprintf(stderr, "penalty: name=%s id=%i  value=%1.2f  len=%i\n", PEN->name, PEN->id, value, PEN->len) ;
	if (PEN==NULL)
		return 0 ;
	if (PEN->cache && value>0 && value<=PEN->max_len)
		return PEN->cache[value] ;

	REAL d_value = (REAL) value ;
	switch (PEN->transform)
	{
	case T_LINEAR:
		break ;
	case T_LOG:
		d_value = log(d_value) ;
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
	INT i=0 ;
	
	REAL ret ;

	i = math.fast_find_range(PEN->limits, PEN->len, value) ;
	if (i==-1)
		ret=PEN->penalties[0] ;
	else if (i==PEN->len-1)
		ret=PEN->penalties[PEN->len-1] ;
	else
	{
		INT i_smaller = i ;
		INT i_larger  = i+1 ;
		
		
		if (PEN->limits[i_larger]==PEN->limits[i_smaller])
			ret=(PEN->penalties[i_smaller]/2+PEN->penalties[i_larger]/2) ;
		else
			ret= (PEN->penalties[i_smaller]*(PEN->limits[i_larger]-value) + 
				  PEN->penalties[i_larger]*(value-PEN->limits[i_smaller]))/
				(PEN->limits[i_larger]-PEN->limits[i_smaller]) ;
	}
	//fprintf(stderr, "penalty: id=%i  value=%1.2f  ret=%1.2f\n", PEN->id, value, ret) ;
	
	return ret ;
}

#endif
