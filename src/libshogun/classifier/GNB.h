/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef GNB_H_
#define GNB_H_

#include "Classifier.h"

namespace shogun {

/** @brief
 *
 */

class CGNB : public CClassifier
{
public:

	CGNB();
	CGNB(CFeatures* train_examples, CLabels* train_labels);
	virtual ~CGNB();

	virtual bool train(CFeatures* data = NULL);
	virtual CLabels* classify();
	virtual CLabels* classify(CFeatures* data);
	virtual float64_t classify_example(int32_t idx);

	virtual bool load(FILE* srcfile);
	virtual bool save(FILE* dstfile);

	virtual inline const char* get_name() const { return "Gaussian Naive Bayes"; };
	virtual inline EClassifierType get_classifier_type() { return CT_GNB; };
};

}

#endif /* GNB_H_ */
