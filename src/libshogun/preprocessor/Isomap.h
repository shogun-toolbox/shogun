/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ISOMAP_H_
#define ISOMAP_H_
#ifdef HAVE_LAPACK

#include "preprocessor/ClassicMDS.h"
#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "distance/Distance.h"

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class Isomap
 */
class CIsomap: public CClassicMDS
{
public:

	/* constructor */
	CIsomap();

	/* destructor */
	virtual ~CIsomap();

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* data);

	/** cleanup
	 *
	 */
	virtual void cleanup();

	/** apply preproc to feature matrix
	 *
	 */
	virtual float64_t* apply_to_feature_matrix(CFeatures* f);

	/** apply preproc to feature vector
	 *
	 */
	virtual float64_t* apply_to_feature_vector(float64_t* f, int32_t &len);

	/** get name */
	virtual inline const char* get_name() const { return "ISOMAP"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_ISOMAP; };

};

}

#endif /* HAVE_LAPACK */
#endif /* ISOMAP_H_ */
