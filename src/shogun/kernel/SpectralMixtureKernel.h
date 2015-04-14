/*
 * This software is distributed under BSD Clause 3 license (see LICENSE file).
 *
 * Written (W) 2015 Esben Soerig
 */

#ifndef SPECTRALMIXTUREKERNEL_H
#define SPECTRALMIXTUREKERNEL_H

#include <shogun/kernel/DotKernel.h>

namespace shogun
{
	class CDotFeatures;

/** @brief The spectral mixture kernel
 * 
 * TODO
 *
 */

class CSpectralMixtureKernel: public CDotKernel
{
	public:
		/** default constructor */
		CSpectralMixtureKernel();

		/** constructor
		 *
		 * The num_components argument specifies how many spectral mixture 
		 * components to initialize. The gaussian spectral mixture components are
		 * initilized stochastically using a standard initialization scheme  
		 * based on metrics from the left-hand and right-hand side features.
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param num_components number of spectral mixture components
		 * @param size cache size. Default value: 10
		 */
		CSpectralMixtureKernel(CDotFeatures* l, CDotFeatures* r,
							   int32_t num_components, int32_t size=10);

		/** constructor with specific parameter initializations.
		*
		* The number of mixture components is implicitly specified by the
		* lengths of the weights, means, and stddeviations vectors (which must 
		* be of equal length).
		*
		* @param weights vector of weights for each mixture component
		* @param means vector of means for each mixture component
		* @param stddeviations vector of standard deviations for each mixture component
		* @param size cache size. Default value: 10
		*/
		CSpectralMixtureKernel(const SGVector<float64_t>& weights, 
							   const SGVector<float64_t>& means,
							   const SGVector<float64_t>& stddeviations,
							   int32_t size=10);

		virtual bool init(CFeatures* lhs, CFeatures* rhs) { return CDotKernel::init(lhs, rhs); }

		/** (re)initialize spectral mixture components.
		*
		* @param num_components number of spectral mixture components to initialize.
		* @return if initializing was successful
		*/
		virtual bool init_components(int32_t num_components);

		/** return what type of kernel we are
		 *
		 * @return kernel type K_SPECTRAL_MIXTURE
		 */
		virtual EKernelType get_kernel_type() { return (EKernelType)540; }

		/** return the kernel's name
		 *
		 * @return name SpectralMixtureKernel
		 */
		virtual const char* get_name() const { return "SpectralMixtureKernel"; }

		/** return the kernel's number of spectral mixture components
		 *
		 * @return number of spectral mixture components
		 */
		virtual int32_t get_num_components() const;

		/** specify the spectral mixture components explicitly. 
		 *
		 * @param weights vector with a weight for each component
		 * @param means vector with a mean for each component
		 * @param stddeviations vector with a standard deviation for each component
		 */
		virtual void set_component_parameters(const SGVector<float64_t>& weights, 
											  const SGVector<float64_t>& means,
											  const SGVector<float64_t>& stddeviations);

		/** return the respective weight for each spectral mixture component.
		 *
		 * @return vector with the weight of each mixture component.
		 */
		virtual SGVector<float64_t> get_weights() const { return m_weights; }

		/** return the respective means for each spectral mixture component.
		 *
		 * @return vector with the mean of each mixture component.
		 */
		virtual SGVector<float64_t> get_means() const { return m_means; }

		/** return the respective standard deviations for each spectral mixture
		 *  component.
		 *
		 * @return vector with the standard deviation of each mixture component.
		 */
		virtual SGVector<float64_t> get_stddeviations() const { return m_stddeviations; }

		/** return derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		virtual SGMatrix<float64_t> get_parameter_gradient(
				const TParameter* param, index_t index);

	private:

		void init();

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

		/** Compute the vector of each feature dotted with itself
		 *
		 * @param features features to square
		 * @return vector of each dot product
		 */
		SGVector<float64_t> get_squared_features(CDotFeatures* features);

		/** Compute matrix of squared distances between lhs and rhs features
		 *
		 * @return square matrix of squared distances. Element (i, j) is the
		 *         squared distance between feature vectors i and and j.
		 */
		SGMatrix<float64_t> get_square_distance_matrix();

		/** Helper function for computing the derivative with respect to a weight
		 *
		 * @param squared_distances matrix of squared distances between features.
		 *                          The result is stored in this matrix.
		 * @param index_t index of weight to compute the derivative with respect to.
		 */
		void compute_weight_derivative(SGMatrix<float64_t>& squared_distances, index_t index);

		/** Helper function for computing the derivative with respect to a mean
		 *
		 * @param squared_distances matrix of squared distances between features.
		 *                          The result is stored in this matrix.
		 * @param index_t index of mean to compute the derivative with respect to.
		 */
		void compute_mean_derivative(SGMatrix<float64_t>& squared_distances, index_t index);

		/** Helper function for computing the derivative with respect to a stddeviation
		 *
		 * @param squared_distances matrix of squared distances between features.
		 *                          The result is stored in this matrix.
		 * @param index_t index of standard deviation to compute the derivative with respect to.
		 */
		void compute_stddeviation_derivative(SGMatrix<float64_t>& squared_distances, index_t index);

		/** weights for each spectral mixture component */
		SGVector<float64_t> m_weights;
		/** means of each (gaussian) spectral mixture component */
		SGVector<float64_t> m_means;
		/** standard deviations of each (gaussian) spectral mixture component */
		SGVector<float64_t> m_stddeviations;
};
}
#endif /* _SPECTRALMIXTUREKERNEL_H_ */
