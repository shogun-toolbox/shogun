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
	bool init_kernel(CHAR* param);
	bool init_kernel_tree(CHAR* param);
	bool delete_kernel_tree(CHAR* param);
	bool load_kernel_init(CHAR* param);
	bool save_kernel_init(CHAR* param);
	bool save_kernel(CHAR* param);

	bool is_initialized() { return initialized ; } ;

 protected:
	CKernel* kernel;
	CGUI* gui ;
	bool initialized;
	
};
#endif
