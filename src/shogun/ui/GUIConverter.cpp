/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <ui/GUIConverter.h>
#include <ui/SGInterface.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>

#include <converter/LocallyLinearEmbedding.h>
#include <converter/HessianLocallyLinearEmbedding.h>
#include <converter/LocalTangentSpaceAlignment.h>
#include <converter/NeighborhoodPreservingEmbedding.h>
#include <converter/LaplacianEigenmaps.h>
#include <converter/LocalityPreservingProjections.h>
#include <converter/DiffusionMaps.h>
#include <converter/LinearLocalTangentSpaceAlignment.h>
#include <converter/MultidimensionalScaling.h>
#include <converter/Isomap.h>
#include <converter/EmbeddingConverter.h>
#include <converter/ica/Jade.h>

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
#ifdef HAVE_EIGEN3
	m_converter = new CLocallyLinearEmbedding();
	((CLocallyLinearEmbedding*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_neighborhoodpreservingembedding(int32_t k)
{
#ifdef HAVE_EIGEN3
	m_converter = new CNeighborhoodPreservingEmbedding();
	((CNeighborhoodPreservingEmbedding*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_localtangentspacealignment(int32_t k)
{
#ifdef HAVE_EIGEN3
	m_converter = new CLocalTangentSpaceAlignment();
	((CLocalTangentSpaceAlignment*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_linearlocaltangentspacealignment(int32_t k)
{
#ifdef HAVE_EIGEN3
	m_converter = new CLinearLocalTangentSpaceAlignment();
	((CLinearLocalTangentSpaceAlignment*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_hessianlocallylinearembedding(int32_t k)
{
#ifdef HAVE_EIGEN3
	m_converter = new CLocallyLinearEmbedding();
	((CHessianLocallyLinearEmbedding*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_laplacianeigenmaps(int32_t k, float64_t width)
{
#ifdef HAVE_EIGEN3
	m_converter = new CLaplacianEigenmaps();
	((CLaplacianEigenmaps*)m_converter)->set_k(k);
	((CLaplacianEigenmaps*)m_converter)->set_tau(width);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_localitypreservingprojections(int32_t k, float64_t width)
{
#ifdef HAVE_EIGEN3
	m_converter = new CLocalityPreservingProjections();
	((CLocalityPreservingProjections*)m_converter)->set_k(k);
	((CLocalityPreservingProjections*)m_converter)->set_tau(width);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_diffusionmaps(int32_t t, float64_t width)
{
#ifdef HAVE_EIGEN3
	m_converter = new CDiffusionMaps();
	((CDiffusionMaps*)m_converter)->set_t(t);
	((CDiffusionMaps*)m_converter)->set_kernel(new CGaussianKernel(100,width));
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_isomap(int32_t k)
{
#ifdef HAVE_EIGEN3
	m_converter = new CIsomap();
	((CIsomap*)m_converter)->set_k(k);
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_multidimensionalscaling()
{
#ifdef HAVE_EIGEN3
	m_converter = new CMultidimensionalScaling();
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
	return true;
}

bool CGUIConverter::create_jade()
{
#ifdef HAVE_EIGEN3
	m_converter = new CJade();
#else
	SG_ERROR("Requires EIGEN3 to be enabled at compile time\n")
#endif
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

