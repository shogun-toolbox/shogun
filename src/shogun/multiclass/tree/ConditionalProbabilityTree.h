/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef CONDITIONALPROBABILITYTREE_H__
#define CONDITIONALPROBABILITYTREE_H__

#include <shogun/multiclass/tree/TreeMachine.h>

namespace shogun
{

class CConditionalProbabilityTree: public CTreeMachine
{
public:
    /** constructor */
	CConditionalProbabilityTree() {}

    /** destructor */
	virtual ~CConditionalProbabilityTree() {}

    /** get name */
    virtual const char* get_name() const { return "ConditionalProbabilityTree"; }

private:
	/** to prevent compile error of class_list.cpp */
	virtual void __placeholder__()=0;
};

} /* shogun */ 

#endif /* end of include guard: CONDITIONALPROBABILITYTREE_H__ */

