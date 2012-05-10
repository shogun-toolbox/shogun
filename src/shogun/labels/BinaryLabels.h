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
#include <shogun/mathematics/CMath.h>

namespace shogun
{

struct BinaryLabelElem
{
    typedef int32_t   label_t;
    typedef float64_t real_label_t;

    label_t      get_label() const { return CMath::sign(val); }
    real_label_t get_real_label() const { return val; }
    float64_t    get_confidence() const { return CMath::abs(val); }
    void         set_label(const real_label_t& label) { val = label; }
    void         set_confidence(float64_t confidence)
    {
        val = CMath::abs(confidence) * CMath::sign(val);
    }

    label_t val;
};

class CBinaryLabels: public CLabelsImpl<BinaryLabelElem>
{
public:
    typedef CLabelsImpl<BinaryLabelElem> base_t;

    /** constructor */
    CBinaryLabels():base_t() {}

    
    /** constructor
     *
     * @param num number of labels
     */
    CBinaryLabels(int32_t num):base_t(num) { }

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

