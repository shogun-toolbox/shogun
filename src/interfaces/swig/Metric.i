/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(LMNNStatistics) CLMNNStatistics;
%rename(LMNN) CLMNN;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/metric/LMNN.h>
