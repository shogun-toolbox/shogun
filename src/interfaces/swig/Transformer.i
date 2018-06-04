/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

/* These functions return new Objects */
%newobject shogun::CTransformer::transform(CFeatures*, bool inplace=true);
%newobject shogun::CTransformer::inverse_transform(CFeatures*, bool inplace=true);

/* Remove C Prefix */
%rename(Transformer) CTransformer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/transformer/Transformer.h>

