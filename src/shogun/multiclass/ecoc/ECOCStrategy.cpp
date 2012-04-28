/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/ecoc/ECOCStrategy.h>

using namespace shogun;

CECOCStrategy::CECOCStrategy()
{
    init();
}

CECOCStrategy::CECOCStrategy(CECOCEncoder *encoder)
    :m_encoder(encoder)
{
    init();
}

void CECOCStrategy::init()
{
    SG_REF(m_encoder);
    SG_ADD((CSGObject **)&m_encoder, "encoder", "ECOC Encoder", MS_NOT_AVAILABLE);
}

CECOCStrategy::~CECOCStrategy()
{
    SG_UNREF(m_encoder);

    m_codebook.destroy_matrix();
}

void CECOCStrategy::train_start(CLabels *orig_labels, CLabels *train_labels)
{
    CMulticlassStrategy::train_start(orig_labels, train_labels);

    m_codebook = m_encoder->create_codebook(m_num_classes);
}

bool CECOCStrategy::train_has_more()
{
    return m_train_iter < m_codebook.num_cols;
}

SGVector<int32_t> CECOCStrategy::train_prepare_next()
{
    SGVector<int32_t> subset(m_orig_labels->get_num_labels());
    int32_t tot=0;
    for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
    {
        int32_t label = m_orig_labels->get_int_label(i);
        switch (m_codebook(label, m_train_iter))
        {
        case -1:
            m_train_labels->set_label(i, -1);
            subset[tot++]=i;
            break;
        case 1:
            m_train_labels->set_label(i, 1);
            subset[tot++]=i;
            break;
        default:
            // 0 means ignore
            break;
        }
    }

    CMulticlassStrategy::train_prepare_next();
    return SGVector<int32_t>(subset.vector, tot);
}

int32_t CECOCStrategy::decide_label(const SGVector<float64_t> &outputs)
{
    // TODO: implement this with a decoder
    return 0;
}

int32_t CECOCStrategy::get_num_machines()
{
    return m_codebook.num_cols;
}
