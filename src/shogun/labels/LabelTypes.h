/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Yuyu Zhang, Viktor Gal, Thoralf Klein, 
 *          Fernando Iglesias, Soeren Sonnenburg, Jiaolong Xu
 */
#ifndef _LABEL_TYPES__H__
#define _LABEL_TYPES__H__

#include <shogun/lib/config.h>

enum ELabelType
{
	/// binary labels +1/-1
	LT_BINARY = 0,
	/// multi-class labels 0,1,...
	LT_MULTICLASS = 1,
	/// real valued labels (e.g. for regression, classifier outputs)
	LT_REGRESSION = 3,
	/// structured labels (e.g. sequences, trees) used in Structured Output
	/// problems
	LT_STRUCTURED = 4,
	/// latent latent labels
	LT_LATENT = 5,
	/// sparse label class for multilabel classification (sets of labels)
	LT_SPARSE_MULTILABEL = 6,
};
#endif
