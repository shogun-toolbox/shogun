/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */
 
%rename(EmbeddingConverter) CEmbeddingConverter;
%rename(LocallyLinearEmbedding) CLocallyLinearEmbedding;
%rename(NeighborhoodPreservingEmbedding) CNeighborhoodPreservingEmbedding;
%rename(LocalTangentSpaceAlignment) CLocalTangentSpaceAlignment;
%rename(LinearLocalTangentSpaceAlignment) CLinearLocalTangentSpaceAlignment;
%rename(HessianLocallyLinearEmbedding) CHessianLocallyLinearEmbedding;
%rename(KernelLocallyLinearEmbedding) CKernelLocallyLinearEmbedding;
%rename(DiffusionMaps) CDiffusionMaps;
%rename(LaplacianEigenmaps) CLaplacianEigenmaps;
%rename(LocalityPreservingProjections) CLocalityPreservingProjections;
%rename(MultidimensionalScaling) CMultidimensionalScaling;
%rename(Isomap) CIsomap;
%rename(StochasticProximityEmbedding) CStochasticProximityEmbedding;
%rename(FactorAnalysis) CFactorAnalysis;
%rename(TDistributedStochasticNeighborEmbedding) CTDistributedStochasticNeighborEmbedding;
%rename(ManifoldSculpting) CManifoldSculpting;
%rename(HashedDocConverter) CHashedDocConverter;
%rename(ICAConverter) CICAConverter;
%rename(Jade) CJade;
%rename(SOBI) CSOBI;
%rename(FFSep) CFFSep;
%rename(JediSep) CJediSep;
%rename(UWedgeSep) CUWedgeSep;
%rename(FastICA) CFastICA;

%newobject shogun::CEmbeddingConverter::apply;
%newobject shogun::*::embed_kernel;
%newobject shogun::*::embed_distance;

%include <shogun/converter/Converter.h>
%include <shogun/converter/EmbeddingConverter.h>
%include <shogun/converter/LocallyLinearEmbedding.h>
%include <shogun/converter/NeighborhoodPreservingEmbedding.h>
%include <shogun/converter/LocalTangentSpaceAlignment.h>
%include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
%include <shogun/converter/HessianLocallyLinearEmbedding.h>
%include <shogun/converter/KernelLocallyLinearEmbedding.h>
%include <shogun/converter/DiffusionMaps.h>
%include <shogun/converter/LaplacianEigenmaps.h>
%include <shogun/converter/LocalityPreservingProjections.h>
%include <shogun/converter/MultidimensionalScaling.h>
%include <shogun/converter/Isomap.h>
%include <shogun/converter/StochasticProximityEmbedding.h>
%include <shogun/converter/FactorAnalysis.h>
%include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
%include <shogun/converter/ManifoldSculpting.h>
%include <shogun/converter/HashedDocConverter.h>
%include <shogun/converter/ica/ICAConverter.h>
%include <shogun/converter/ica/Jade.h>
%include <shogun/converter/ica/SOBI.h>
%include <shogun/converter/ica/FFSep.h>
%include <shogun/converter/ica/JediSep.h>
%include <shogun/converter/ica/UWedgeSep.h>
%include <shogun/converter/ica/FastICA.h>
