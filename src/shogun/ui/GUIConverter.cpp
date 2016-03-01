/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/ui/GUIConverter.h>
#include <shogun/ui/SGInterface.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/converter/HessianLocallyLinearEmbedding.h>
#include <shogun/converter/LocalTangentSpaceAlignment.h>
#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/converter/LaplacianEigenmaps.h>
#include <shogun/converter/LocalityPreservingProjections.h>
#include <shogun/converter/DiffusionMaps.h>
#include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/converter/Isomap.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/converter/ica/Jade.h>

using namespace shogun;

CGUIConverter::CGUIConverter(CSGInterface* ui)
: CSGObject(), m_ui(ui)
{
	m_converter = NULL;
}

CGUIConverter::~CGUIConverter()
{
	SG_UNREF(m_converter);
}

bool CGUIConverter::create_locallylinearembedding(int32_t k)
{
	m_converter = new CLocallyLinearEmbedding();
	((CLocallyLinearEmbedding*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_neighborhoodpreservingembedding(int32_t k)
{
	m_converter = new CNeighborhoodPreservingEmbedding();
	((CNeighborhoodPreservingEmbedding*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_localtangentspacealignment(int32_t k)
{
	m_converter = new CLocalTangentSpaceAlignment();
	((CLocalTangentSpaceAlignment*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_linearlocaltangentspacealignment(int32_t k)
{
	m_converter = new CLinearLocalTangentSpaceAlignment();
	((CLinearLocalTangentSpaceAlignment*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_hessianlocallylinearembedding(int32_t k)
{
	m_converter = new CLocallyLinearEmbedding();
	((CHessianLocallyLinearEmbedding*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_laplacianeigenmaps(int32_t k, float64_t width)
{
	m_converter = new CLaplacianEigenmaps();
	((CLaplacianEigenmaps*)m_converter)->set_k(k);
	((CLaplacianEigenmaps*)m_converter)->set_tau(width);
	return true;
}

bool CGUIConverter::create_localitypreservingprojections(int32_t k, float64_t width)
{
	m_converter = new CLocalityPreservingProjections();
	((CLocalityPreservingProjections*)m_converter)->set_k(k);
	((CLocalityPreservingProjections*)m_converter)->set_tau(width);
	return true;
}

bool CGUIConverter::create_diffusionmaps(int32_t t, float64_t width)
{
	m_converter = new CDiffusionMaps();
	((CDiffusionMaps*)m_converter)->set_t(t);
	((CDiffusionMaps*)m_converter)->set_kernel(new CGaussianKernel(100,width));
	return true;
}

bool CGUIConverter::create_isomap(int32_t k)
{
	m_converter = new CIsomap();
	((CIsomap*)m_converter)->set_k(k);
	return true;
}

bool CGUIConverter::create_multidimensionalscaling()
{
	m_converter = new CMultidimensionalScaling();
	return true;
}

bool CGUIConverter::create_jade()
{
	m_converter = new CJade();
	return true;
}

CDenseFeatures<float64_t>* CGUIConverter::apply()
{
	if (!m_converter)
		SG_ERROR("No converter created")
	return (CDenseFeatures<float64_t>*)m_converter->apply(m_ui->ui_features->get_train_features());
}

CDenseFeatures<float64_t>* CGUIConverter::embed(int32_t target_dim)
{
	if (!m_converter)
		SG_ERROR("No converter created")
	((CEmbeddingConverter*)m_converter)->set_target_dim(target_dim);
	return ((CEmbeddingConverter*)m_converter)->embed(m_ui->ui_features->get_train_features());
}

