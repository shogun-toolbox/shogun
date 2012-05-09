/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef BINARYLABELS_H__
#define BINARYLABELS_H__

#include <shogun/labels/LabelsImpl.h>

namespace shogun
{

class CBinaryLabels: public CLabelsImpl<int32_t, float64_t>
{
public:
    /** constructor */
    CBinaryLabels():CLabelsImpl<int32_t, float64_t>() {}

    
    /** constructor
     *
     * @param num number of labels
     */
    CBinaryLabels(int32_t num):CLabelsImpl<int32_t, float64_t>(num) { }

    /** destructor */
    virtual ~CBinaryLabels() {}

    /** get name */
    virtual const char* get_name() const { return "BinaryLabels"; }

    /** get number of classes */
    virtual int32_t get_num_classes() const { return 2; }

    /** get label type */
    virtual ELabelType get_type() const { return LT_BINARY; }
};

} /* shogun */ 

#endif /* end of include guard: BINARYLABELS_H__ */

