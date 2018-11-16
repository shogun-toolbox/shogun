/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Thoralf Klein, 
 *          Fernando Iglesias, Bjoern Esser
 */

#include <shogun/labels/LatentLabels.h>

using namespace shogun;

CLatentLabels::CLatentLabels()
	: CLabels()
{
	init();
}

CLatentLabels::CLatentLabels(int32_t num_samples)
	: CLabels()
{
	init();
	m_latent_labels = new CDynamicObjectArray(num_samples);
	SG_REF(m_latent_labels);
}

CLatentLabels::CLatentLabels(CLabels* labels)
	: CLabels()
{
	init();
	set_labels(labels);

	int32_t num_labels = 0;
	if (m_labels)
		num_labels = m_labels->get_num_labels();

	m_latent_labels = new CDynamicObjectArray(num_labels);
	SG_REF(m_latent_labels);
}

CLatentLabels::~CLatentLabels()
{
	SG_UNREF(m_latent_labels);
	SG_UNREF(m_labels);
}

void CLatentLabels::init()
{
	SG_ADD((CSGObject**) &m_latent_labels, "m_latent_labels", "The latent labels");
	SG_ADD((CSGObject**) &m_labels, "m_labels", "The labels");
	m_latent_labels = NULL;
	m_labels = NULL;
}

CDynamicObjectArray* CLatentLabels::get_latent_labels() const
{
	SG_REF(m_latent_labels);
	return m_latent_labels;
}

CData* CLatentLabels::get_latent_label(int32_t idx)
{
	ASSERT(m_latent_labels != NULL)
	if (idx < 0 || idx >= get_num_labels())
		SG_ERROR("Out of index!\n")

	return (CData*) m_latent_labels->get_element(idx);
}

void CLatentLabels::add_latent_label(CData* label)
{
	ASSERT(m_latent_labels != NULL)
	m_latent_labels->push_back(label);
}

bool CLatentLabels::set_latent_label(int32_t idx, CData* label)
{
	if (idx < get_num_labels())
	{
		return m_latent_labels->set_element(label, idx);
	}
	else
	{
		return false;
	}
}

bool CLatentLabels::is_valid() const
{
	return m_latent_labels != nullptr;
}

void CLatentLabels::ensure_valid(const char* context)
{
	REQUIRE(is_valid(), "Empty labels provided!\n");
}

int32_t CLatentLabels::get_num_labels() const
{
	if (!m_latent_labels || !m_labels)
		return 0;
	int32_t num_labels = m_latent_labels->get_num_elements();

	ASSERT(num_labels == m_labels->get_num_labels())

	return num_labels;
}

void CLatentLabels::set_labels(CLabels* labels)
{
	SG_REF(labels);
	SG_UNREF(m_labels);
	m_labels = labels;
}

CLabels* CLatentLabels::get_labels() const
{
	SG_REF(m_labels);
	return m_labels;
}

