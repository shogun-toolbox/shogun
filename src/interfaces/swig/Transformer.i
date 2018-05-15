/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

/* Remove C Prefix */
%rename(CTransformer) CTransformer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/transformer/Transformer.h>

