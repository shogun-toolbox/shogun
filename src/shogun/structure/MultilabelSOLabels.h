/*
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNO General Public License as published by the Free
 * Software Foundation; either version 3 of the license, or (at your option)
 * any later version.
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#ifndef _MULTILABEL_SO_LABELS__H__
#define _MULTILABEL_SO_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class CMultilabelSOLabels;

/** @brief Class CSparseLabel to be used in the application of Structured
 * Output (SO) learning to Multilabel classification.*/
class CSparseLabel : public CStructuredData
{
    public:
        /** data type */
        STRUCTURED_DATA_TYPE(SDT_SPARSE_LABEL);

        /** default constructor */
        CSparseLabel() { }

        /** constructor
         *
         * @param label sparse label
         */
        CSparseLabel(SGVector<int32_t> label) : CStructuredData(), m_label(label) { }

        /** destructor */
        ~CSparseLabel() { }

        /** helper method used to specialize a base class instance
         *
         * @param base_data its dynamic type must be CSparseLabel
         */
        static CSparseLabel* obtain_from_generic(CStructuredData* base_data)
        {
            if (base_data->get_structured_data_type() == SDT_SPARSE_LABEL)
                return (CSparseLabel*) base_data;
            else
                SG_SERROR("base_data must be of dynamic type CSparseLabel\n");

            return NULL;
        }

        /** @return name of SGSerializable */
        virtual const char* get_name() const { return "SparseLabel"; }

        SGVector<int32_t> get_data() const { return m_label; }

    protected:
        /** sparse label */
        SGVector<int32_t> m_label;
}; /* class CSparseLabel */

/** @brief Class CMultilabelSOLabels used in the application of Structured
 * Output (SO) learning to Multilabel Classification. Labels are subsets
 * of {0, 1, ..., num_classes-1}. Each of the label if of type CSparseLabel and
 * all of them are stored in a CDynamicObjectArray.
 */
class CMultilabelSOLabels : public CStructuredLabels
{
    public:
        /** default constructor */
        CMultilabelSOLabels();

        /** constructor
         *
         * @param num_classes number of (binary) class assignment per label
         */
        CMultilabelSOLabels(int32_t num_classes);

        /** constructor
         *
         * @param num_labels number of labels
         * @param num_classes number of (binary) class assignment per label
         */
        CMultilabelSOLabels(int32_t num_labels, int32_t num_classes);

        /** destructor */
        ~CMultilabelSOLabels();

        /** @return name of the SGSerializable */
        virtual const char* get_name() const { return "MultilabelSOLabels"; }

        /** @return number of stored labels */
        virtual int32_t get_num_labels() const;

        /** @return number of classes (per label) */
        virtual int32_t get_num_classes() const;

        /** set sparse labels
         *
         * @param labels list of sparse labels
         */
        void set_sparse_labels(SGVector<int32_t>* labels);

        /** set sparse assignment for j-th label
         *
         * @param j label index
         * @param label sparse label
         */
        void set_sparse_label(int32_t j, SGVector<int32_t> label);

        /** Convert sparse labels to dense. The dense vector would be {d_true;
         * d_false}^dense_dim. Indices in sparse would be marked "d_true",
         * everything else "d_false".
         *
         * @param label sparse label to convert
         * @param dense_dim dense dimension
         * @param d_true marker for "true" labels
         * @param d_false marker for "false" labels
         *
         * @return SGVector<D> dense vector of dimension dense_dim
         */
        static SGVector<float64_t> to_dense(CStructuredData* label,
                int32_t dense_dim, float64_t d_true, float64_t d_false);

    private:
        int32_t m_num_classes;
        void init();

}; /* class CMultilabelSOLabels */

} /* namespace shogun */

#endif /* _MULTILABEL_SO_LABELS__H__ */
