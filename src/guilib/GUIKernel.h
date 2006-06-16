/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
#ifdef USE_SVMLIGHT
	bool resize_kernel_cache(CHAR* param);
#endif
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
