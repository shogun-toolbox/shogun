#ifndef _OPTIMIZABLEKERNEL_H___
#define _OPTIMIZABLEKERNEL_H___

#include "lib/common.h"

#include <assert.h>
class CKernel ;

class COptimizableKernel
{
public: 
	COptimizableKernel() ;
	virtual ~COptimizableKernel() ;

	virtual bool init_optimization(INT count, INT *IDX, REAL * weights) ; 
	virtual void delete_optimization()  ;
	virtual REAL compute_optimized(INT idx) ;
	
	bool get_is_initialized() { return initialized ; } ;
	void set_is_initialized(bool init) { initialized=init ; } ;

	static bool is_optimizable(CKernel *k) ;
protected:
	bool initialized ;
} ;


#endif
