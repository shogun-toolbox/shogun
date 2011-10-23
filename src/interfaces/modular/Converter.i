/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011
 */
 
%rename(EmbeddingConverter) CEmbeddingConverter;
%rename(LocallyLinearEmbedding) CLocallyLinearEmbedding;
%rename(NeighborhoodPreservingEmbedding) CNeighborhoodPreservingEmbedding;
%rename(LocalTangentSpaceAlignment) CLocalTangentSpaceAlignment;
%rename(LinearLocalTangentSpaceAlignment) CLinearLocalTangentSpaceAlignment;
%rename(HessianLocallyLinearEmbedding) CHessianLocallyLinearEmbedding;
%rename(KernelLocallyLinearEmbedding) CKernelLocallyLinearEmbedding;
%rename(KernelLocalTangentSpaceAlignment) CKernelLocalTangentSpaceAlignment;
%rename(DiffusionMaps) CDiffusionMaps;
%rename(LaplacianEigenmaps) CLaplacianEigenmaps;
%rename(MultidimensionalScaling) CMultidimensionalScaling;
%rename(Isomap) CIsomap;

%newobject shogun::CEmbeddingConverter::apply;

%include <shogun/converter/Converter.h>
%include <shogun/converter/EmbeddingConverter.h>
%include <shogun/converter/LocallyLinearEmbedding.h>
%include <shogun/converter/NeighborhoodPreservingEmbedding.h>
%include <shogun/converter/LocalTangentSpaceAlignment.h>
%include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
%include <shogun/converter/HessianLocallyLinearEmbedding.h>
%include <shogun/converter/KernelLocallyLinearEmbedding.h>
%include <shogun/converter/KernelLocalTangentSpaceAlignment.h>
%include <shogun/converter/DiffusionMaps.h>
%include <shogun/converter/LaplacianEigenmaps.h>
%include <shogun/converter/MultidimensionalScaling.h>
%include <shogun/converter/Isomap.h>

