/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer,
 *          Heiko Strathmann, Fernando Iglesias
 */

#include <shogun/converter/DiffusionMaps.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/lib/config.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

DiffusionMaps::DiffusionMaps() :
		EmbeddingConverter()
{
	m_t = 10;
	m_width = 1.0;
	set_distance(std::make_shared<EuclideanDistance>());

	init();
}

void DiffusionMaps::init()
{
	SG_ADD(&m_t, "t", "number of steps", ParameterProperties::HYPER);
	SG_ADD(&m_width, "width", "gaussian kernel width", ParameterProperties::HYPER);
}

DiffusionMaps::~DiffusionMaps()
{
}

void DiffusionMaps::set_t(int32_t t)
{
	m_t = t;
}

int32_t DiffusionMaps::get_t() const
{
	return m_t;
}

void DiffusionMaps::set_width(float64_t width)
{
	m_width = width;
}

float64_t DiffusionMaps::get_width() const
{
	return m_width;
}

const char* DiffusionMaps::get_name() const
{
	return "DiffusionMaps";
};

std::shared_ptr<Features> DiffusionMaps::transform(std::shared_ptr<Features> features, bool inplace)
{
	ASSERT(features)
	// shorthand for simplefeatures

	// compute distance matrix
	ASSERT(m_distance)
	m_distance->init(features,features);
	auto embedding = embed_distance(m_distance);
	m_distance->cleanup();

	return std::static_pointer_cast<Features>(embedding);
}

std::shared_ptr<DenseFeatures<float64_t>> DiffusionMaps::embed_distance(std::shared_ptr<Distance> distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_timesteps = m_t;
	parameters.gaussian_kernel_width = m_width;
	parameters.method = SHOGUN_DIFFUSION_MAPS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance.get();
	return tapkee_embed(parameters);
}
