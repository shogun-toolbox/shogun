/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Thoralf Klein, 
 *          Fernando Iglesias, Bjoern Esser
 */

#include <shogun/labels/LatentLabels.h>

#include <utility>

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
	m_latent_labels.reserve(num_samples);
	
}

LatentLabels::LatentLabels(std::shared_ptr<Labels> labels)
	: Labels()
{
	init();
	set_labels(std::move(labels));

	int32_t num_labels = 0;
	if (m_labels)
		num_labels = m_labels->get_num_labels();

	m_latent_labels.reserve(num_labels);
	
}

LatentLabels::~LatentLabels()
{
	
	
}

void LatentLabels::init()
{
	SG_ADD(&m_latent_labels, "m_latent_labels", "The latent labels");
	SG_ADD((std::shared_ptr<SGObject>*) &m_labels, "m_labels", "The labels");
	m_latent_labels.clear();
	m_labels = NULL;
}

std::vector<std::shared_ptr<Data>> LatentLabels::get_latent_labels() const
{
	
	return m_latent_labels;
}

std::shared_ptr<Data> LatentLabels::get_latent_label(int32_t idx)
{
	if (idx < 0 || idx >= get_num_labels())
		error("Out of index!");

	return m_latent_labels[idx];
}

void LatentLabels::add_latent_label(const std::shared_ptr<Data>& label)
{
	m_latent_labels.push_back(label);
}

bool LatentLabels::set_latent_label(int32_t idx, std::shared_ptr<Data> label)
{
	if (idx >= 0 && idx < get_num_labels())
	{
		m_latent_labels[idx] = std::move(label);
		return true;
	}
	else
	{
		return false;
	}
}

bool LatentLabels::is_valid() const
{
	return true;
}

void LatentLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Empty labels provided!");
}

int32_t LatentLabels::get_num_labels() const
{
	if (!m_labels)
		return 0;
	int32_t num_labels = m_latent_labels.size();

	ASSERT(num_labels == m_labels->get_num_labels())

	return num_labels;
}

void LatentLabels::set_labels(std::shared_ptr<Labels> labels)
{
	
	
	m_labels = std::move(labels);
}

std::shared_ptr<Labels> LatentLabels::get_labels() const
{
	
	return m_labels;
}

