/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(Regression) CRegression;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>
