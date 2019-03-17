/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

%rename(EmbeddingConverter) CEmbeddingConverter;

%newobject shogun::*::embed_kernel;
%newobject shogun::*::embed_distance;

%include <shogun/converter/Converter.h>
%include <shogun/converter/EmbeddingConverter.h>
