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
	INT min_len ;
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

inline REAL lookup_penalty(const struct penalty_struct *PEN, INT p_value)
{
	if (PEN==NULL)
		return 0 ;
	if (PEN->cache!=NULL && (p_value>=0) && (p_value<=PEN->max_len))
		return PEN->cache[p_value] ;

	if ((p_value<PEN->min_len) || (p_value>PEN->max_len))
		return -math.INFTY ;
	
	REAL d_value = (REAL) p_value ;
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

	INT idx = 0 ;
	REAL ret ;
	for (INT i=0; i<PEN->len; i++)
		if (PEN->limits[i]<=d_value)
			idx++ ;
	
	if (idx==0)
		ret=PEN->penalties[0] ;
	else if (idx==PEN->len)
		ret=PEN->penalties[PEN->len-1] ;
	else
	{
		ret = (PEN->penalties[idx]*(d_value-PEN->limits[idx-1]) + PEN->penalties[idx-1]*
			   (PEN->limits[idx]-d_value)) / (PEN->limits[idx]-PEN->limits[idx-1]) ;  
	}
	//if (p_value>=30 && p_value<150)
	//fprintf(stderr, "%s %i(%i) -> %1.2f\n", PEN->name, p_value, idx, ret) ;
	
	
	return ret ;
}

#endif
