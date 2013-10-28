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
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

CECOCStrategy::CECOCStrategy() : CMulticlassStrategy()
{
    init();
}

CECOCStrategy::CECOCStrategy(CECOCEncoder *encoder, CECOCDecoder *decoder)
    : CMulticlassStrategy()
{
    init();
    m_encoder=encoder;
    m_decoder=decoder;
    SG_REF(m_encoder);
    SG_REF(decoder);
}

void CECOCStrategy::init()
{
    m_encoder=NULL;
    m_decoder=NULL;

    SG_ADD((CSGObject **)&m_encoder, "encoder", "ECOC Encoder", MS_NOT_AVAILABLE);
    SG_ADD((CSGObject **)&m_decoder, "decoder", "ECOC Decoder", MS_NOT_AVAILABLE);
}

CECOCStrategy::~CECOCStrategy()
{
    SG_UNREF(m_encoder);
    SG_UNREF(m_decoder);
}

void CECOCStrategy::train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels)
{
    CMulticlassStrategy::train_start(orig_labels, train_labels);

    m_codebook = m_encoder->create_codebook(m_num_classes);
}

bool CECOCStrategy::train_has_more()
{
    return m_train_iter < m_codebook.num_rows;
}

SGVector<int32_t> CECOCStrategy::train_prepare_next()
{
    SGVector<int32_t> subset(m_orig_labels->get_num_labels(), false);
    int32_t tot=0;
    for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
    {
        int32_t label = ((CMulticlassLabels*) m_orig_labels)->get_int_label(i);
        switch (m_codebook(m_train_iter, label))
        {
        case -1:
            ((CBinaryLabels*) m_train_labels)->set_label(i, -1);
            subset[tot++]=i;
            break;
        case 1:
            ((CBinaryLabels*) m_train_labels)->set_label(i, 1);
            subset[tot++]=i;
            break;
        default:
            // 0 means ignore
            break;
        }
    }

    CMulticlassStrategy::train_prepare_next();
    return SGVector<int32_t>(subset.vector, tot, true);
}

int32_t CECOCStrategy::decide_label(SGVector<float64_t> outputs)
{
    return m_decoder->decide_label(outputs, m_codebook);
}

int32_t CECOCStrategy::get_num_machines()
{
    return m_codebook.num_cols;
}
