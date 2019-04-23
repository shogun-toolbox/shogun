/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Thoralf Klein, 
 *          Fernando Iglesias, Bjoern Esser
 */

#include <shogun/labels/LatentLabels.h>

using namespace shogun;

LatentLabels::LatentLabels()
	: Labels()
{
	init();
}

LatentLabels::LatentLabels(int32_t num_samples)
	: Labels()
{
	init();
	m_latent_labels = std::make_shared<DynamicObjectArray>(num_samples);
	
}

LatentLabels::LatentLabels(std::shared_ptr<Labels> labels)
	: Labels()
{
	init();
	set_labels(labels);

	int32_t num_labels = 0;
	if (m_labels)
		num_labels = m_labels->get_num_labels();

	m_latent_labels = std::make_shared<DynamicObjectArray>(num_labels);
	
}

LatentLabels::~LatentLabels()
{
	
	
}

void LatentLabels::init()
{
	SG_ADD((std::shared_ptr<SGObject>*) &m_latent_labels, "m_latent_labels", "The latent labels");
	SG_ADD((std::shared_ptr<SGObject>*) &m_labels, "m_labels", "The labels");
	m_latent_labels = NULL;
	m_labels = NULL;
}

std::shared_ptr<DynamicObjectArray> LatentLabels::get_latent_labels() const
{
	
	return m_latent_labels;
}

std::shared_ptr<Data> LatentLabels::get_latent_label(int32_t idx)
{
	ASSERT(m_latent_labels != NULL)
	if (idx < 0 || idx >= get_num_labels())
		error("Out of index!");

	return std::static_pointer_cast<Data>( m_latent_labels->get_element(idx));
}

void LatentLabels::add_latent_label(std::shared_ptr<Data> label)
{
	ASSERT(m_latent_labels != NULL)
	m_latent_labels->push_back(label);
}

bool LatentLabels::set_latent_label(int32_t idx, std::shared_ptr<Data> label)
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

bool LatentLabels::is_valid() const
{
	return m_latent_labels != nullptr;
}

void LatentLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Empty labels provided!");
}

int32_t LatentLabels::get_num_labels() const
{
	if (!m_latent_labels || !m_labels)
		return 0;
	int32_t num_labels = m_latent_labels->get_num_elements();

	ASSERT(num_labels == m_labels->get_num_labels())

	return num_labels;
}

void LatentLabels::set_labels(std::shared_ptr<Labels> labels)
{
	
	
	m_labels = labels;
}

std::shared_ptr<Labels> LatentLabels::get_labels() const
{
	
	return m_labels;
}

