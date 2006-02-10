#ifndef __GUIKERNEL__H
#define __GUIKERNEL__H

#include "kernel/Kernel.h"
#include "features/Features.h"

class CGUI ;

class CGUIKernel
{
 public:
	CGUIKernel(CGUI*);
	~CGUIKernel();

	CKernel* get_kernel();
	bool set_kernel(CHAR* param);
	CKernel* create_kernel(CHAR* params);
	bool init_kernel(CHAR* param);
	bool init_kernel_optimization(CHAR* param);
	bool delete_kernel_optimization(CHAR* param);
	bool load_kernel_init(CHAR* param);
	bool save_kernel_init(CHAR* param);
	bool save_kernel(CHAR* param);

	bool clean_kernel(CHAR* param);
	bool resize_kernel_cache(CHAR* param);
	bool set_optimization_type(CHAR* param);
	bool add_kernel(CHAR* param);
	bool del_kernel(CHAR* param);
	bool is_initialized() { return initialized ; } ;

 protected:
	CKernel* kernel;
	CGUI* gui ;
	bool initialized;
};
#endif
