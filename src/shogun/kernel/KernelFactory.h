/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef KERNELFACTORY_H__
#define KERNELFACTORY_H__

#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CKernelFactory: public CSGObject
{
public:
    /** constructor */
	CKernelFactory() {}

    /** destructor */
	virtual ~CKernelFactory() {}

    /** get name */
    virtual const char* get_name() const { return "KernelFactory"; }

	/** construct a *new* kernel */
	virtual CKernel *make_kernel()=0;
};

} /* shogun */ 

#endif /* end of include guard: KERNELFACTORY_H__ */

