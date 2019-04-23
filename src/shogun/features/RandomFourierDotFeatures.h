/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Yuyu Zhang, Bjoern Esser
 */

#ifndef _RANDOMFOURIER_DOTFEATURES__H__
#define _RANDOMFOURIER_DOTFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/RandomKitchenSinksDotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
template <class ST> class DenseFeatures;
class DotFeatures;

/** names of kernels that can be approximated currently */
enum KernelName
{
	/** approximate gaussian kernel
	 *	expects one parameter to be specified :
	 *		kernel width
	 */
	GAUSSIAN,

	/** not specified */
	NOT_SPECIFIED
};

/** @brief This class implements the random fourier features for the DotFeatures
 *  framework.
 *  Basically upon the object creation it computes the random coefficients, namely w and b,
 *  that are needed for this method and then every time a vector is required it is computed
 *  based on the following formula z(x) = sqrt(2/D) * cos(w'*x + b), where D is the number
 *  of samples that are used.
 *
 *  For more detailed information you can take a look at this source:
 *  i) Random Features for Large-Scale Kernel Machines - Ali Rahimi and Ben Recht
 */
class RandomFourierDotFeatures : public RandomKitchenSinksDotFeatures
{
public:

	/** default constructor */
	RandomFourierDotFeatures();

	/** constructor that creates new random coefficients, basedon the kernel specified and the parameters
	 * of the kernel.
	 *
	 * @param features the dense features to use as a base
	 * @param D the number of random fourier samples to draw / dimensionality of new feature space
	 * @param kernel_name the name of the kernel to approximate
	 * @param params kernel parameters (see kernel's description in KernelName to see what each kernel expects)
	 */
	RandomFourierDotFeatures(std::shared_ptr<DotFeatures> features, int32_t D, KernelName kernel_name,
			SGVector<float64_t> params);

	/** constructor that uses the specified random coefficients.
	 *
	 * @param features the dense features to use as a base
	 * @param D the number of random fourier samples to draw / dimensionality of new feature space
	 * @param kernel_name the name of the kernel to approximate
	 * @param params kernel parameters (see kernel's description in KernelName to see what each kernel expects)
	 * @param coeff pre-computed random coefficients to use
	 */
	RandomFourierDotFeatures(std::shared_ptr<DotFeatures> features, int32_t D, KernelName kernel_name,
			SGVector<float64_t> params, SGMatrix<float64_t> coeff);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	RandomFourierDotFeatures(std::shared_ptr<File> loader);

	/** copy constructor */
	RandomFourierDotFeatures(const RandomFourierDotFeatures& orig);

	/** duplicate */
	virtual std::shared_ptr<Features> duplicate() const;

	/** destructor */
	virtual ~RandomFourierDotFeatures();

	/** @return object name */
	virtual const char* get_name() const;

protected:

	/** subclass must override this to perform any operations
	 * on the dot result between a feature vector and a parameter vector w
	 *
	 * @param dot_result the result of the dot operation
	 * @param par_idx the idx of the parameter vector
	 * @return the (optionally) modified result
	 */
	virtual float64_t post_dot(float64_t dot_result, index_t par_idx) const;

	/** Generates a random parameter vector, subclasses must override this
	 *
	 * @return a random parameter vector
	 */
	virtual SGVector<float64_t> generate_random_parameter_vector();

private:
	void init(KernelName kernel_name, SGVector<float64_t> params);

private:
	/** the kernel to approximate */
	KernelName kernel;

	/** The parameters of the kernel to approximate */
	SGVector<float64_t> kernel_params;

	/** norm const */
	float64_t constant;
};
}

#endif // _RANDOMFOURIER_DOTFEATURES__H__
