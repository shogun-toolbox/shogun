/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef CLASSICMDS_H_
#define CLASSICMDS_H_
#ifdef HAVE_LAPACK

#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "distance/Distance.h"

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class ClassicMDS used to perform classic eigenvector
 * 	multidimensional scaling.
 *
 * 	Description is given at p.261 (Section 12.1) of
 * 	Borg, I., & Groenen, P. J. F. (2005).
 * 	Modern multidimensional scaling: Theory and applications. Springer.
 *
 */
class CClassicMDS: public CSimplePreprocessor<float64_t>
{
public:

	/* constructor */
	CClassicMDS();

	/* destructor */
	virtual ~CClassicMDS();

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
	virtual inline const char* get_name() const { return "CLASSICMDS"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_CLASSICMDS; };

	/** setter for target dimension
	 * @param dim target dimension
	 */
	void inline set_target_dim(int32_t dim)
	{
		ASSERT(dim>0);
		m_target_dim = dim;
	}

	/** getter for target dimension
	 * @return target dimension
	 */
	int32_t inline get_target_dim()
	{
		return m_target_dim;
	}

protected:

	/* target dim */
	int32_t m_target_dim;
};

}

#endif /* HAVE_LAPACK */
#endif /* CLASSICMDS_H_ */
