#ifndef __PENALTY_INFO_H__
#define __PENALTY_INFO_H__

#include "lib/common.h"
#include "lib/Mathmatics.h"

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

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
	struct penalty_struct *next_pen ;
	char * name ;
	bool use_svm ;
} ;

void init_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct_array(struct penalty_struct *PEN, INT len) ;

#ifdef HAVE_MATLAB
struct penalty_struct * read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT &P) ;
#endif

REAL lookup_penalty(const struct penalty_struct *PEN, INT p_value, REAL svm_value, bool follow_next=true) ;

#endif
