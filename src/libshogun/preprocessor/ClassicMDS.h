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
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "features/Features.h"
#include "distance/Distance.h"

namespace shogun
{

class CFeatures;
#include "preprocessor/DimensionReductionPreprocessor.h"
class CDistance;

/** @brief the class ClassicMDS used to perform classic eigenvector
 * 	multidimensional scaling.
 *
 * 	Description is given at p.261 (Section 12.1) of
 * 	Borg, I., & Groenen, P. J. F. (2005).
 * 	Modern multidimensional scaling: Theory and applications. Springer.
 *
 */
class CClassicMDS: public CDimensionReductionPreprocessor
{
public:

	/* constructor */
	CClassicMDS();

	/* destructor */
	virtual ~CClassicMDS();

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 *
	 */
	virtual void cleanup();

	/** apply preproc to distance
	 *
	 */
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance);

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** get name */
	virtual inline const char* get_name() const { return "CLASSICMDS"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_CLASSICMDS; };

	/** get last embedding eigenvectors */
	SGVector<float64_t> get_eigenvalues() const
	{
		return m_eigenvalues;
	}

protected:

	/** positive eigenvalues of last call in descending order */
	SGVector<float64_t> m_eigenvalues;

	/** apply preproc to distance
	 *
	 */
	SGMatrix<float64_t> embed_by_distance(CDistance* distance);

};

}

#endif /* HAVE_LAPACK */
#endif /* CLASSICMDS_H_ */
