/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::LMNNStatistics)
%shared_ptr(shogun::LMNN)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/metric/LMNN.h>
