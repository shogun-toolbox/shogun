/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Bjoern Esser, Viktor Gal
 */

#ifndef _DIRECTOR_LATENT_MODEL_H_
#define _DIRECTOR_LATENT_MODEL_H_

#include <shogun/latent/LatentModel.h>
#include <shogun/lib/config.h>
#ifdef USE_SWIG_DIRECTORS
namespace shogun
{

class LatentModel;

#define IGNORE_IN_CLASSLIST
/**
 * @brief Class CDirectorLatentModel that represents the application specific model
 * with latent variable svm in target interface language. It is a base class
 * that needs to be extended with real implementations before using.
 *
 * @see LatentModel
 */
IGNORE_IN_CLASSLIST class DirectorLatentModel : public LatentModel
{
	public:
		/** default constructor */
		DirectorLatentModel();

		/** destructor */
		~DirectorLatentModel() override;

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		int32_t get_dim() const override;

		/** Calculate the PSI vectors for all features
		 *
		 * @return PSI vectors
		 */
		std::shared_ptr<DotFeatures> get_psi_feature_vectors() override;

		/** User defined \f$h^{*} = argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
		 * This function has to be defined the user as it is applications specific, since
		 * it depends on the user defined latent feature and latent label.
		 *
		 * @param w weight vector
		 * @param idx index of the example
		 * @return returns \f$h^{*}\f$ for the given example
		 */
		std::shared_ptr<Data> infer_latent_variable(const SGVector<float64_t>& w, index_t idx) override;

		/** Calculates \f$argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
		 * The default implementaiton calculates the argmax_h only on the positive examples.
		 *
		 * @param w weight vector (cutting plane) supplied by the underlying optimizer.
		 */
		void argmax_h(const SGVector<float64_t>& w) override;

		/** @return name of SGSerializable */
		const char* get_name() const override { return "DirectorLatentModel"; }

}; /* class CDirectorLatentModel */
} /* namespace shogun */
#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTOR_LATENT_MODEL_H_ */

