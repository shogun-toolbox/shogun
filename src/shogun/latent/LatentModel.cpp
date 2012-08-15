/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/latent/LatentModel.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CLatentModel::CLatentModel()
	: m_features(NULL), m_labels(NULL)
{
	register_parameters();
}

CLatentModel::CLatentModel(CLatentFeatures* feats, CLatentLabels* labels)
	: m_features(feats), m_labels(labels)
{
	register_parameters();
	SG_REF(m_features);
	SG_REF(m_labels);
}

CLatentModel::~CLatentModel()
{
	SG_UNREF(m_labels);
	SG_UNREF(m_features);
}

int32_t CLatentModel::get_num_vectors() const
{
	return m_features->get_num_vectors();
}

void CLatentModel::set_labels(CLatentLabels* labs)
{
	SG_UNREF(m_labels);
	SG_REF(labs);
	m_labels = labs;
}

CLatentLabels* CLatentModel::get_labels() const
{
	SG_REF(m_labels);
	return m_labels;
}

void CLatentModel::set_features(CLatentFeatures* feats)
{
	SG_UNREF(m_features);
	SG_REF(feats);
	m_features = feats;
}

void CLatentModel::argmax_h(const SGVector<float64_t>& w)
{
	int32_t num = get_num_vectors();
	CBinaryLabels* y = CBinaryLabels::obtain_from_generic(m_labels->get_labels());
	ASSERT(num > 0);
	ASSERT(num == m_labels->get_num_labels());
	

	// argmax_h only for positive examples
	for (int32_t i = 0; i < num; ++i)
	{
		if (y->get_label(i) == 1)
		{
			// infer h and set it for the argmax_h <w,psi(x,h)>
			CData* latent_data = infer_latent_variable(w, i);
			m_labels->set_latent_label(i, latent_data);
		}
	}
}

void CLatentModel::register_parameters()
{
	m_parameters->add((CSGObject**) &m_features, "features", "Latent features");
	m_parameters->add((CSGObject**) &m_labels, "labels", "Latent labels");
}

