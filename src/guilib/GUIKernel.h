#ifndef __GUIKERNEL__H
#define __GUIKERNEL__H

#include "kernel/Kernel.h"

class CGUI ;

class CGUIKernel
{
 public:
	CGUIKernel(CGUI*);
	~CGUIKernel();

	CKernel* get_kernel();
	bool set_kernel(char* param);
	bool init_kernel(char* param);
	bool load_kernel_init(char* param);
	bool save_kernel_init(char* param);
	bool save_kernel(char* param);

 protected:
	CKernel* kernel;
	CGUI* gui ;
};
#endif
