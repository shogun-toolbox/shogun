/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LABELS_H__
#define LABELS_H__

namespace shogun
{

/** label type */
enum ELabelType
{
    LT_BINARY = 0,
    LT_REGRESSION = 10,
    LT_MULTICLASS = 20,
};

/** pure virtual base class for Labels.
 */
class CLabels: public CSGObject
{
public:
    /** constructor */
    CLabels() {}

    /** destructor */
    virtual ~CLabels() {}

    /** get name */
    virtual const char* get_name() const { return "Labels"; }

    /** get label type */
    virtual ELabelType get_type() const = 0;

    /** get number of classes */
    virtual int32_t get_num_classes() const = 0;

    /** get number of labels */
    virtual int32_t get_num_labels() const = 0;

    /** adds a subset of indices on top of the current subsets. 
     * Calls subset_changed_post() afterwards.
     *
     * @param subset subset of indices to add
     */
    virtual void add_subset(SGVector<index_t> subset) = 0;

    /** remove the last added subset from the subset stack, if existing.
     * Calls subset_changed_post() afterwards
     */
    virtual void remove_subset() = 0;

    /** removes all subsets.
     * Calls subset_changed_post() afterwards */
    virtual void remove_all_subsets() = 0;
};

} /* shogun */ 

#endif /* end of include guard: LABELS_H__ */

