/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef GENERICKERNELFACTORY_H__
#define GENERICKERNELFACTORY_H__

#include <shogun/kernel/KernelFactory.h>

namespace shogun
{

template <typename KernelType> 
	class CGenericKernelFactory
	:public CKernelFactory
{
public:
    /** constructor */
	CGenericKernelFactory() {}

    /** destructor */
	virtual ~CGenericKernelFactory() {}

    /** get name */
    virtual const char* get_name() const { return "GenericKernelFactory"; }

	/** construct a *new* kernel */
	virtual CKernel *make_kernel()
	{
		CKernel *kernel = new KernelType();
		SG_REF(kernel);
		return kernel;
	}
};

} /* shogun */ 

#endif /* end of include guard: GENERICKERNELFACTORY_H__ */

