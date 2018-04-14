/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Loss renames */
%rename(LossFunction) CLossFunction;
%rename(HingeLoss) CHingeLoss;
%rename(LogLoss) CLogLoss;
%rename(LogLossMargin) CLogLossMargin;
%rename(SmoothHingeLoss) CSmoothHingeLoss;
%rename(SquearedHingeLoss) CSquaredHingeLoss;
%rename(SquaredLoss) CSquaredLoss;

/* Loss includes */
%include <shogun/loss/LossFunction.h>
%include <shogun/loss/HingeLoss.h>
%include <shogun/loss/LogLoss.h>
%include <shogun/loss/LogLossMargin.h>
%include <shogun/loss/SmoothHingeLoss.h>
%include <shogun/loss/SquaredHingeLoss.h>
%include <shogun/loss/SquaredLoss.h>
