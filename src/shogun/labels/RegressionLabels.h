/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef REGRESSIONLABELS_H__
#define REGRESSIONLABELS_H__

#include <shogun/labels/LabelsImpl.h>

namespace shogun
{

struct RegressionLabelElem
{
    typedef float64_t label_t;
    typedef float64_t real_label_t;

    label_t      get_label() const { return val; }
    real_label_t get_real_label() const { return val; }
    float64_t    get_confidence() const { return 1; }
    void         set_label(const real_label_t& label) { val = label; }
    void         set_confidence(float64_t confidence) { }

    label_t val;
};

class CRegressionLabels: public CLabelsImpl<float64_t, float64_t>
{
public:
    typedef CLabelsImpl<RegressionLabelElem> base_t;

    /** constructor */
    CRegressionLabels():base_t() {}

    /** constructor
     *
     * @param num number of labels
     */
    CRegressionLabels(int32_t num):base_t(num) { }

    /** destructor */
    virtual ~CRegressionLabels() {}

    /** get name */
    virtual const char* get_name() const { return "RegressionLabels"; }

    /** get number of classes.
     *
     * Note there are "infinite" number of classes in an regression problem, but we 
     * return 0 here since we don't have inf in int32_t.
     */
    virtual int32_t get_num_classes() const { return 0; }

    /** get label type */
    virtual ELabelType get_type() const { return LT_REGRESSION; }
};

} /* shogun */ 

#endif /* end of include guard: REGRESSIONLABELS_H__ */

