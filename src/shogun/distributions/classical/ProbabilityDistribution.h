/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifndef PROBABILITYDISTRIBUTION_H
#define PROBABILITYDISTRIBUTION_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

template <class T> class SGVector;

/** @brief A base class for representing n-dimensional probability distribution
 * over the real numbers (64bit) for which various statistics can be computed
 * and which can be sampled.
 */
class CProbabilityDistribution: public CSGObject
{
public:
	/** Default constructor */
	CProbabilityDistribution();

	/** Constructur that sets the distribution's dimension */
	CProbabilityDistribution(int32_t dimension);

	/** Destructor */
	virtual ~CProbabilityDistribution();

	/** Samples from the distribution multiple times
	 *
	 * @param num_samples number of samples to generate
	 * @param pre_samples a matrix of pre-samples that might be used for
	 * sampling. For example, a matrix with standard normal samples for the
	 * CGaussianDistribution. For reproducible results. Ignored by default.
	 * @return matrix with samples (column vectors)
	 */
	virtual SGMatrix<float64_t> sample(int32_t num_samples,
			SGMatrix<float64_t> pre_samples=SGMatrix<float64_t>()) const;

	/** Samples from the distribution once. Wrapper method. No pre-sample
	 * passing is possible with this method.
	 *
	 * @return vector with single sample
	 */
	virtual SGVector<float64_t> sample() const;

	/** Computes the log-pdf for all provided samples
	 *
	 * @param samples samples to compute log-pdf of (column vectors)
	 * @return vector with log-pdfs of given samples
	 */
	virtual SGVector<float64_t> log_pdf_multiple(SGMatrix<float64_t> samples) const;

	/** Computes the log-pdf for a single provided sample. Wrapper method which
	 * calls log_pdf_multiple
	 *
	 * @param sample_vec sample_vec to compute log-pdf for
	 * @return log-pdf of the given sample
	 */
	virtual float64_t log_pdf(SGVector<float64_t> sample_vec) const;

	/** @return name of the SGSerializable */
	virtual const char* get_name() const=0;

    /** Computes the minimum and maximum for 1000 sample points
     * list of tuples with
     * minimum and maximum for each dimension
     *
     * @return list of tuples with
     * minimum and maximum for each dimension
     */
     virtual SGMatrix<float64_t> get_plotting_bounds() const;

    /** Compute points for propsal distributition, which might for example
     * be used in Metropolis_Hastings
     *
     * @param num_samples Number of Samples required
     * @return Array of sample points
     */
     virtual SGVector<float64_t> get_proposal_points(int32_t num_samples) const;

    /** Compute points for acceptance distributition, which might for example
     * be used in Metropolis_Hastings
     *
     * @param num_samples Number of Samples required
     * @return Array of sample points
     */
     virtual SGVector<float64_t> get_acceptance_points(int32_t num_samples) const;

    /** Compute Samples for taking log-likelihood
     *
     * @param num_samples Number of Samples taken
     * @return Array of sample points
     */
     virtual SGVector<float64_t> get_log_likelihood_samples(int32_t
                                                            num_samples) const;

    /** Compute log partial derivatives for maximum log likelihood function
     *
     * @param sample_vec Array of samples taken already
     * @return Array of computed derivatives
     */
     virtual SGVector<float64_t> get_log_partial_derivatives(SGVector<float64_t>
                                                             sample_vec) const;

    /** Compute maximum log likelihood
     *
     * @param partial_derivatives_vec Array of derivatives
     * @return maximum log-likelihood
     */
     virtual float64_t get_maximum_log_likelihood(SGVector<float64_t>
                                                             partial_derivatives_vec) const;

private:

	/** Initialses and registers parameters */
	void init();

protected:
	/** Dimension of the distribution */
	int32_t m_dimension;
};

}

#endif // PROBABILITYDISTRIBUTION_H
