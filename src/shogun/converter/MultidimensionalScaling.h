/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef MULTIDIMENSIONALSCALING_H_
#define MULTIDIMENSIONALSCALING_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/EmbeddingConverter.h>
#include <features/Features.h>
#include <distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class Multidimensionalscaling is used to perform
 * multidimensional scaling (capable of landmark approximation
 * if requested).
 *
 * Description of classical embedding is given on p.261 (Section 12.1) of
 * Borg, I., & Groenen, P. J. F. (2005).
 * Modern multidimensional scaling: Theory and applications. Springer.
 *
 * Description of landmark MDS approximation is given in
 *
 * Sparse multidimensional scaling using landmark points
 * V De Silva, J B Tenenbaum (2004) Technology, p. 1-4
 *
 * Note that target dimension should be set with reasonable value
 * (using set_target_dim). In case it is higher than intrinsic
 * dimensionality of the dataset 'extra' features of the output
 * might be inconsistent (essentially, according to zero or
 * negative eigenvalues). In this case a warning is fired.
 *
 * It is possible to apply multidimensional scaling to any
 * given distance using apply_to_distance_matrix method.
 * By default euclidean distance is used (with parallel
 * instance replaced by preprocessor's one).
 *
 * Faster landmark approximation is parallel using posix threads.
 * As for choice of landmark number it should be at least 3 for
 * proper triangulation. For reasonable embedding accuracy greater
 * values (30%-50% of total examples number) is pretty good for the
 * most tasks.
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','mds');
 *
 */
class CMultidimensionalScaling: public CEmbeddingConverter
{
public:

	/* constructor */
	CMultidimensionalScaling();

	/* destructor */
	virtual ~CMultidimensionalScaling();

	/** apply preprocessor to CDistance
	 * @param distance (should be approximate euclidean for consistent result)
	 * @return new features with distance similar to given as much as possible
	 */
	virtual CDenseFeatures<float64_t>* embed_distance(CDistance* distance);

	/** apply preprocessor to feature matrix,
	 * changes feature matrix to the one having target dimensionality
	 * @param features features which feature matrix should be processed
	 * @return new feature matrix
	 */
	virtual CFeatures* apply(CFeatures* features);

	/** get name */
	const char* get_name() const;

	/** get last embedding eigenvectors
	 * @return vector with last eigenvalues
	 */
	SGVector<float64_t> get_eigenvalues() const;

	/** set number of landmarks
	 * should be lesser than number of examples and greater than 3
	 * for consistent embedding as triangulation is used
	 * @param num number of landmark to be set
	 */
	void set_landmark_number(int32_t num);

	/** get number of landmarks
	 * @return current number of landmarks
	 */
	int32_t get_landmark_number() const;

	/** setter for landmark parameter
	 * @param landmark true if landmark embedding should be used
	 */
	void set_landmark(bool landmark);

	/** getter for landmark parameter
	 * @return true if landmark embedding is used
	 */
	bool get_landmark() const;

/// HELPERS
protected:

	/** default initialization */
	virtual void init();

/// FIELDS
protected:

	/** last embedding eigenvalues */
	SGVector<float64_t> m_eigenvalues;

	/** use landmark approximation? */
	bool m_landmark;

	/** number of landmarks */
	int32_t m_landmark_number;

};

}
#endif /* HAVE_EIGEN3 */
#endif /* MULTIDIMENSIONALSCALING_H_ */
