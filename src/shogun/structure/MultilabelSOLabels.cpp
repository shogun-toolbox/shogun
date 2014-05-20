/*
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNO General Public License as published by the Free
 * Software Foundation; either version 3 of the license, or (at your option)
 * any later version.
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#include <shogun/structure/MultilabelSOLabels.h>

using namespace shogun;

CMultilabelSOLabels::CMultilabelSOLabels() : CStructuredLabels(), m_num_classes(0)
{
}

CMultilabelSOLabels::CMultilabelSOLabels(int32_t num_classes)
    : CStructuredLabels(0), m_num_classes(num_classes)
{
    init();
}

CMultilabelSOLabels::CMultilabelSOLabels(int32_t num_labels, int32_t num_classes)
    : CStructuredLabels(num_labels), m_num_classes(num_classes)
{
    init();
}

void CMultilabelSOLabels::init()
{
    SG_ADD(&m_num_classes, "num_classes", "Number of (binary) classes per label",
            MS_NOT_AVAILABLE);
}

CMultilabelSOLabels::~CMultilabelSOLabels()
{
}

void CMultilabelSOLabels::set_sparse_label(int32_t j, SGVector<int32_t> label)
{
    CStructuredLabels::set_label(j, new CSparseLabel(label));
}

void CMultilabelSOLabels::set_sparse_labels(SGVector<int32_t>* labels)
{
    for (int i=0; i<get_num_labels(); i++)
        CStructuredLabels::set_label(i, new CSparseLabel(labels[i]));
}

int32_t CMultilabelSOLabels::get_num_labels() const
{
    if (m_labels == NULL)
        return 0;
    else
        return m_labels->get_array_size();
}

int32_t CMultilabelSOLabels::get_num_classes() const
{
    return m_num_classes;
}

SGVector<float64_t> CMultilabelSOLabels::to_dense(CStructuredData* label,
        int32_t dense_dim, float64_t d_true, float64_t d_false)
{
    CSparseLabel* slabel = CSparseLabel::obtain_from_generic(label);
    SGVector<int32_t> sparse = slabel->get_data();
    SGVector<float64_t> dense(dense_dim);
    dense.set_const(d_false);
    for (int32_t i=0; i<sparse.vlen; i++)
    {
        int32_t index = sparse[i];
        REQUIRE(index < dense_dim, "class index exceed length of dense vector");
        dense[index] = d_true;
    }
    return dense;
}
