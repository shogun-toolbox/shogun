#include "gui/GUI.h"
#include "gui/TextGUI.h"
extern CTextGUI* gui;

#include "lib/config.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"

#include <stdio.h>
#include <string.h>

#include "lib/io.h"

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

#include "distributions/hmm/penalty_info.h"

void init_penalty_struct(struct penalty_struct &PEN)
{
	PEN.limits=NULL ;
	PEN.penalties=NULL ;
	PEN.id=-1 ;
	PEN.next_pen=NULL ;
	PEN.transform = T_LINEAR ;
	PEN.name = NULL ;
	PEN.max_len=0 ;
	PEN.min_len=0 ;
	PEN.cache=NULL ;
	PEN.use_svm=0 ;
}

void init_penalty_struct_cache(struct penalty_struct &PEN)
{
	if (PEN.cache || PEN.use_svm)
		return ;
		
	REAL* cache=new REAL[PEN.max_len+1] ;
	if (cache)
	{
		REAL input_value ;
		for (INT i=0; i<=PEN.max_len; i++)
			if (i<PEN.min_len)
				cache[i] = -CMath::INFTY ;
			else
				cache[i] = lookup_penalty(&PEN, i, 0, false,input_value) ;
		PEN.cache = cache ;
	}
}

void delete_penalty_struct(struct penalty_struct &PEN)
{
	if (PEN.id!=-1)
	{
		delete[] PEN.limits ;
		delete[] PEN.penalties ;
		delete[] PEN.name ;
		delete[] PEN.cache ;
	}
}

void delete_penalty_struct_array(struct penalty_struct *PEN, INT len)
{
	for (int i=0; i<len; i++)
		delete_penalty_struct(PEN[i]) ;
	delete[] PEN ;
}


#ifdef HAVE_MATLAB
struct penalty_struct * read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT &P)
{
	P = mxGetN(mx_penalty_info) ;
	
	struct penalty_struct * PEN = new struct penalty_struct[P] ;
	for (INT i=0; i<P; i++)
		 init_penalty_struct(PEN[i]) ;
	
	for (INT i=0; i<P; i++)
	{
		const mxArray* mx_elem = mxGetCell(mx_penalty_info, i) ;
		if (mx_elem==NULL || !mxIsStruct(mx_elem))
		{
			CIO::message(M_ERROR, "empty cell element\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_id_field = mxGetField(mx_elem, 0, "id") ;
		if (mx_id_field==NULL || !mxIsNumeric(mx_id_field) || 
			mxGetN(mx_id_field)!=1 || mxGetM(mx_id_field)!=1)
		{
			CIO::message(M_ERROR, "missing id field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_limits_field = mxGetField(mx_elem, 0, "limits") ;
		if (mx_limits_field==NULL || !mxIsNumeric(mx_limits_field) ||
			mxGetM(mx_limits_field)!=1)
		{
			CIO::message(M_ERROR, "missing limits field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		INT len = mxGetN(mx_limits_field) ;
		
		const mxArray* mx_penalties_field = mxGetField(mx_elem, 0, "penalties") ;
		if (mx_penalties_field==NULL || !mxIsNumeric(mx_penalties_field) ||
			mxGetM(mx_penalties_field)!=1 || mxGetN(mx_penalties_field)!=len)
		{
			CIO::message(M_ERROR, "missing penalties field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_transform_field = mxGetField(mx_elem, 0, "transform") ;
		if (mx_transform_field==NULL || !mxIsChar(mx_transform_field))
		{
			CIO::message(M_ERROR, "missing transform field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_name_field = mxGetField(mx_elem, 0, "name") ;
		if (mx_name_field==NULL || !mxIsChar(mx_name_field))
		{
			CIO::message(M_ERROR, "missing name field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_max_len_field = mxGetField(mx_elem, 0, "max_len") ;
		if (mx_max_len_field==NULL || !mxIsNumeric(mx_max_len_field) ||
			mxGetM(mx_max_len_field)!=1 || mxGetN(mx_max_len_field)!=1)
		{
			CIO::message(M_ERROR, "missing max_len field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_min_len_field = mxGetField(mx_elem, 0, "min_len") ;
		if (mx_min_len_field==NULL || !mxIsNumeric(mx_min_len_field) ||
			mxGetM(mx_min_len_field)!=1 || mxGetN(mx_min_len_field)!=1)
		{
			CIO::message(M_ERROR, "missing min_len field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		const mxArray* mx_use_svm_field = mxGetField(mx_elem, 0, "use_svm") ;
		if (mx_use_svm_field==NULL || !mxIsNumeric(mx_use_svm_field) ||
			mxGetM(mx_use_svm_field)!=1 || mxGetN(mx_use_svm_field)!=1)
		{
			CIO::message(M_ERROR, "missing use_svm field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		INT use_svm = (INT) mxGetScalar(mx_use_svm_field) ;
		//fprintf(stderr, "use_svm_field=%i\n", use_svm) ;
		
		const mxArray* mx_next_id_field = mxGetField(mx_elem, 0, "next_id") ;
		if (mx_next_id_field==NULL || !mxIsNumeric(mx_next_id_field) ||
			mxGetM(mx_next_id_field)!=1 || mxGetN(mx_next_id_field)!=1)
		{
			CIO::message(M_ERROR, "missing next_id field\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		INT next_id = (INT) mxGetScalar(mx_next_id_field)-1 ;
		
		INT id = (INT) mxGetScalar(mx_id_field)-1 ;
		if (i<0 || i>P-1)
		{
			CIO::message(M_ERROR, "id out of range\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		INT max_len = (INT) mxGetScalar(mx_max_len_field) ;
		if (max_len<0 || max_len>1024*1024*100)
		{
			CIO::message(M_ERROR, "max_len out of range\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		PEN[id].max_len = max_len ;

		INT min_len = (INT) mxGetScalar(mx_min_len_field) ;
		if (min_len<0 || min_len>1024*1024*100)
		{
			CIO::message(M_ERROR, "min_len out of range\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		PEN[id].min_len = min_len ;

		if (PEN[id].id!=-1)
		{
			CIO::message(M_ERROR, "penalty id already used\n") ;
			delete_penalty_struct_array(PEN,P) ;
			return NULL ;
		}
		PEN[id].id=id ;
		if (next_id>=0)
			PEN[id].next_pen=&PEN[next_id] ;
		//fprintf(stderr,"id=%i, next_id=%i\n", id, next_id) ;
		
		ASSERT(next_id!=id) ;
		PEN[id].use_svm=use_svm ;
		PEN[id].limits = new REAL[len] ;
		PEN[id].penalties = new REAL[len] ;
		double * limits = mxGetPr(mx_limits_field) ;
		double * penalties = mxGetPr(mx_penalties_field) ;
		
		for (INT i=0; i<len; i++)
		{
			PEN[id].limits[i]=limits[i] ;
			PEN[id].penalties[i]=penalties[i] ;
		}
		PEN[id].len = len ;
		
		char *transform_str = mxArrayToString(mx_transform_field) ;				
		char *name_str = mxArrayToString(mx_name_field) ;				

		if (strcmp(transform_str, "log")==0)
			PEN[id].transform = T_LOG ;
		else if (strcmp(transform_str, "log(+1)")==0)
			PEN[id].transform = T_LOG_PLUS1 ;	
		else if (strcmp(transform_str, "log(+3)")==0)
			PEN[id].transform = T_LOG_PLUS3 ;	
		else if (strcmp(transform_str, "(+3)")==0)
			PEN[id].transform = T_LINEAR_PLUS3 ;	
		else if (strcmp(transform_str, "")==0)
			PEN[id].transform = T_LINEAR ;	
		else
		{
			delete_penalty_struct_array(PEN,P) ;
			mxFree(transform_str) ;
			return NULL ;
		}
		PEN[id].name = new char[strlen(name_str)+1] ;
		strcpy(PEN[id].name, name_str) ;

		init_penalty_struct_cache(PEN[id]) ;

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

REAL lookup_penalty_svm(const struct penalty_struct *PEN, INT p_value, REAL *d_values, bool follow_next, 
						REAL &input_value)
{	
	if (PEN==NULL)
		return 0 ;
	ASSERT(PEN->use_svm>0) ;
	REAL d_value=d_values[PEN->use_svm-1] ;
    input_value = d_value ;
	//fprintf(stderr,"transform=%i, d_value=%1.2f\n", (INT)PEN->transform, d_value) ;
	
	switch (PEN->transform)
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
	
	if (PEN->next_pen && follow_next)
		ret+=lookup_penalty(PEN->next_pen, p_value, d_values, follow_next, input_value);
	
	return ret ;
}

REAL lookup_penalty(const struct penalty_struct *PEN, INT p_value, 
					REAL* svm_values, bool follow_next, REAL &input_value)
{	
	if (PEN==NULL)
		return 0 ;
	if (PEN->use_svm)
		return lookup_penalty_svm(PEN, p_value, svm_values, follow_next, input_value) ;
		
	input_value = (REAL) p_value ;

	if ((p_value<PEN->min_len) || (p_value>PEN->max_len))
		return -CMath::INFTY ;
	
	if (PEN->cache!=NULL && (p_value>=0) && (p_value<=PEN->max_len))
	{
		REAL ret=PEN->cache[p_value] ;
		if (PEN->next_pen && follow_next)
			ret+=lookup_penalty(PEN->next_pen, p_value, svm_values, true, input_value);
		return ret ;
	}
	
	REAL d_value = (REAL) p_value ;
	switch (PEN->transform)
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
	
	if (PEN->next_pen && follow_next)
		ret+=lookup_penalty(PEN->next_pen, p_value, svm_values, true, input_value);

	return ret ;
}
