/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

/* These functions return new Objects */
%newobject shogun::CTransformer::transform(CFeatures*);

/* Remove C Prefix */
%rename(Transformer) CTransformer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/transformer/Transformer.h>

