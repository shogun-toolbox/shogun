#ifndef __GUIKERNEL__H
#define __GUIKERNEL__H

#include "kernel/Kernel.h"

class CGUI ;

class CGUIKernel
{
 public:
	CGUIKernel(CGUI*);
	~CGUIKernel();

 protected:
	CKernel* kernel;
 protected:
	CGUI* gui ;
};
#endif
