#ifndef __GUIKERNEL__H
#define __GUIKERNEL__H

#include "kernel/Kernel.h"

class CGUIKernel
{
public:
	CGUIKernel();
	~CGUIKernel();

protected:
	CKernel* kernel;
};
#endif
