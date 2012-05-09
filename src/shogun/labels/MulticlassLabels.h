/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef MULTICLASSLABELS_H__
#define MULTICLASSLABELS_H__

#include <shogun/labels/LabelsImpl.h>

namespace shogun
{

class CMulticlassLabels: public CLabelsImpl<int32_t, int32_t>
{
public:
    /** constructor */
    CMulticlassLabels():CLabelsImpl<int32_t, int32_t>(), m_num_classes(0) { }

    /** constructor
     *
     * @param num number of labels
     */
    CMulticlassLabels(int32_t num):CLabelsImpl<int32_t, int32_t>(num), m_num_classes(0) { }

    /** destructor */
    virtual ~CMulticlassLabels() {}

    /** get name */
    virtual const char* get_name() const { return "MulticlassLabels"; }

    /** get number of classes.
     */
    virtual int32_t get_num_classes() const 
    { 
        // lazily compute number of classes
        if (m_num_classes == 0)
            m_num_classes = compute_num_classes();

        return m_num_classes; 
    }

    /** set label
     *
     * possible with subset
     *
     * @param idx index of label to set
     * @param label value of label
     * @return if setting was successful
     */
    bool set_label(int32_t idx, const Tstore& label)
    {
        if (CLabelsImpl<int32_t, int32_t>::set_label(idx, label))
        {
            m_num_classes = 0;
            return true;
        }
        return false;
    }

    /** get label type */
    virtual ELabelType get_type() const { return LT_MULTICLASS; }

protected:
    int32_t m_num_classes;

    int32_t compute_num_classes()
    {
        // TODO: implement this
        return 0;
    }
};

} /* shogun */ 

#endif /* end of include guard: MULTICLASSLABELS_H__ */

