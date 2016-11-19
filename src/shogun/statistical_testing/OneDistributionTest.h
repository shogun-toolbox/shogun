/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ONE_DISTRIBUTION_TEST_H_
#define ONE_DISTRIBUTION_TEST_H_

#include <shogun/statistical_testing/HypothesisTest.h>

namespace shogun
{

/**
 * @brief Class OneDistributionTest is the base class for the statistical
 * hypothesis testing with samples from one distributions, \f$mathbf{P}\f$.
 */
class COneDistributionTest : public CHypothesisTest
{
public:
	/** Default constructor */
	COneDistributionTest();

	/** Destrutor */
	virtual ~COneDistributionTest();

	/**
	 * Method that initializes the samples from \f$\mathbf{P}\f$.
	 *
	 * @param samples The CFeatures instance representing the samples
	 * from \f$\mathbf{P}\f$.
	 */
	void set_samples(CFeatures* samples);

	/** @return The samples from \f$\mathbf{P}\f$. */
	CFeatures* get_samples() const;

	/**
	 * Method that initializes the number of samples to be drawn from distribution
	 * \f$\mathbf{P}\f$. Please ensure to call this method if you are intending to
	 * use streaming data generators that generate the samples on the fly. For
	 * other types of features, the number of samples is set internally from the
	 * features object itself, therefore this method should not be used.
	 *
	 * @param num_samples The CFeatures instance representing the samples
	 * from \f$\mathbf{P}\f$.
	 */
	void set_num_samples(index_t num_samples);

	/** @return The number of samples from \f$\mathbf{P}\f$. */
	index_t get_num_samples() const;

	/**
	 * Interface for computing the test-statistic for the hypothesis test.
	 *
	 * @return test statistic for the given data/parameters/methods
	 */
	virtual float64_t compute_statistic()=0;

	/**
	 * Interface for computing the samples under the null-hypothesis.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null()=0;

	/** @return The name of the class */
	virtual const char* get_name() const;
};

}
#endif // ONE_DISTRIBUTION_TEST_H_
