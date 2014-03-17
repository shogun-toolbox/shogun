/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef _LABEL_TYPES__H__
#define _LABEL_TYPES__H__
enum ELabelType
{
	/// binary labels +1/-1
	LT_BINARY = 0,
	/// multi-class labels 0,1,...
	LT_MULTICLASS = 1,
	/// real valued labels (e.g. for regression, classifier outputs)
	LT_REGRESSION = 3,
	/// structured labels (e.g. sequences, trees) used in Structured Output problems
	LT_STRUCTURED = 4,
	/// latent latent labels
	LT_LATENT = 5,
	/// multiple output multiclass
	LT_MULTICLASS_MULTIPLE_OUTPUT = 6,
	/// sparse label class for multilabel classification (sets of labels)
	LT_SPARSE_MULTILABEL = 7,
};
#endif
