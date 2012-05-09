/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LABELSIMPL_H__
#define LABELSIMPL_H__

#include <shogun/lib/SGVector.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{

/**
 * A template implementation of CLabels.
 *
 * There are two template type parameters T, and TStore. T should be able
 * to implicitly cast-able to TStore.
 */
template <typename T, typename TStore>
class CLabelsImpl: public CLabels
{
public:
    /** constructor */
    CLabelsImpl() { init(); }

    /** constructor
     *
     * @param num number of labels
     */
    CLabelsImpl(int32_t num):m_labels(num) { init(); }

    /** destructor */
    virtual ~CLabelsImpl() 
    {
        SG_UNREF(m_subset_stack);
    }

    /** get name */
    virtual const char* get_name() const { return "LabelsImpl"; }

    /** get number of labels */
    virtual int32_t get_num_labels() const 
    {
        return m_subset_stack->has_subset() ? 
            m_subset_stack->get_size() : m_labels.vlen; 
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
        if (idx >= get_num_labels())
            return false;

        int32_t true_idx = m_subset_stack->subset_idx_conversion(idx);
        m_labels[true_idx] = label;
        return true;
    }

    /** get label.
     *
     * possible with subset
     *
     * @param idx index of label to get
     * @return value of label
     */
    T get_label(int32_t idx) const
    {
        return get_real_labels();
    }

    /** get real label.
     *
     * possible with subset
     *
     * @param idx index of the label to get
     * @return value of label
     */
    const TStore& get_real_label(int32_t idx) const
    {
        ASSERT(idx < get_num_labels());
        return m_labels[m_subset_stack->subset_idx_conversion(idx)];
    }

    /** adds a subset of indices on top of the current subsets. 
     * Calls subset_changed_post() afterwards.
     *
     * @param subset subset of indices to add
     */
    virtual void add_subset(SGVector<index_t> subset)
    {
        m_subset_stack->add_subset(subset);
    }

    /** remove the last added subset from the subset stack, if existing.
     * Calls subset_changed_post() afterwards
     */
    virtual void remove_subset()
    {
        m_subset_stack->remove_subset();
    }

    /** removes all subsets.
     * Calls subset_changed_post() afterwards */
    virtual void remove_all_subsets()
    {
        m_subset_stack->remove_all_subset();
    }

protected:
    SGVector<TStore> m_labels;
    CSubsetStack *m_subset_stack;

private:
    void init()
    {
        m_parameters->add(&m_labels, "labels", "The labels");
        m_parameters->add((CSGObject**)&m_subset_stack, "subset_stack", "Subset stack");

        m_subset_stack = new CSubsetStack();
        SG_REF(m_subset_stack);
    }
};

} /* shogun */ 

#endif /* end of include guard: LABELSIMPL_H__ */

