/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shell Hu, Yuyu Zhang, Thoralf Klein,
 *          Bjoern Esser, Sergey Lisitsyn, Viktor Gal
 */

#ifndef _STRUCTURED_OUTPUT_MACHINE__H__
#define _STRUCTURED_OUTPUT_MACHINE__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/structure/SOSVMHelper.h>

namespace shogun
{

class Features;
class Labels;
class LossFunction;
class StructuredLabels;
struct TMultipleCPinfo;

/** The structured empirical risk types, corresponding to different training objectives [1].
 *
 * [1] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural SVMs,
 * Machine Learning Journal, 2009.
 */
enum EStructRiskType
{
	N_SLACK_MARGIN_RESCALING = 0,
	N_SLACK_SLACK_RESCALING = 1,
	ONE_SLACK_MARGIN_RESCALING = 2,
	ONE_SLACK_SLACK_RESCALING = 3,
	CUSTOMIZED_RISK = 4
};

class StructuredModel;

/** TODO doc */
class StructuredOutputMachine : public Machine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_STRUCTURED);

		/** deafult constructor */
		StructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param labs structured labels
		 */
		StructuredOutputMachine(std::shared_ptr<StructuredModel> model, const std::shared_ptr<StructuredLabels>& labs);

		/** destructor */
		virtual ~StructuredOutputMachine();

		/** set structured model
		 *
		 * @param model structured model to set
		 */
		void set_model(std::shared_ptr<StructuredModel> model);

		/** get structured model
		 *
		 * @return structured model
		 */
		std::shared_ptr<StructuredModel> get_model() const;

		/** @return object name */
		virtual const char* get_name() const
		{
			return "StructuredOutputMachine";
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(std::shared_ptr<Labels> lab);

		/** set features
		 *
		 * @param f features
		 */
		void set_features(std::shared_ptr<Features> f);

		/** get features
		 *
		 * @return features
		 */
		std::shared_ptr<Features> get_features() const;

		/** set surrogate loss function
		 *
		 * @param loss loss function to set
		 */
		void set_surrogate_loss(std::shared_ptr<LossFunction> loss);

		/** get surrogate loss function
		 *
		 * @return loss function
		 */
		std::shared_ptr<LossFunction> get_surrogate_loss() const;

		/** computes the value of the risk function and sub-gradient at given point
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @param rtype The type of structured risk
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk(SGVector<float64_t>& subgrad, SGVector<float64_t>& W,
				TMultipleCPinfo* info=0, EStructRiskType rtype = N_SLACK_MARGIN_RESCALING);

		/** @return training progress helper */
		std::shared_ptr<SOSVMHelper> get_helper() const;

		/** set verbose
		 * NOTE that track verbose information including primal objectives,
		 * training errors and duality gaps will make the training 2x or 3x slower.
		 *
		 * @param verbose flag enabling/disabling verbose information
		 */
		void set_verbose(bool verbose);

		/** get verbose
		 *
		 * @return Status of verbose flag (enabled/disabled)
		 */
		bool get_verbose() const;

	protected:
		/** n-slack formulation and margin rescaling
		 *
		 * The value of the risk is evaluated as
		 *
		 * \f[
		 * R({\bf w}) = \sum_{i=1}^{m} \max_{y \in \mathcal{Y}} \left[ \ell(y_i, y)
		 * + \langle {\bf w}, \Psi(x_i, y) - \Psi(x_i, y_i)  \rangle  \right]
		 * \f]
		 *
		 * The subgradient is by Danskin's theorem given as
		 *
		 * \f[
		 * R'({\bf w}) = \sum_{i=1}^{m} \Psi(x_i, \hat{y}_i) - \Psi(x_i, y_i),
		 * \f]
		 *
		 * where \f$ \hat{y}_i \f$ is the most violated label, i.e.
		 *
		 * \f[
		 * \hat{y}_i = \arg\max_{y \in \mathcal{Y}} \left[ \ell(y_i, y)
		 * + \langle {\bf w}, \Psi(x_i, y)  \rangle \right]
		 * \f]
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk_nslack_margin_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info=0);

		/** n-slack formulation and slack rescaling
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk_nslack_slack_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info=0);

		/** 1-slack formulation and margin rescaling
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk_1slack_margin_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info=0);

		/** 1-slack formulation and slack rescaling
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk_1slack_slack_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info=0);

		/** customized risk type
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk_customized_formulation(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info=0);

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** the model that contains the application dependent modules */
		std::shared_ptr<StructuredModel> m_model;

		/** the surrogate loss, for SOSVM, fixed to Hinge loss,
		 * other non-convex losses such as Ramp loss are also applicable,
		 * will be extended in the future
		 */
		std::shared_ptr<LossFunction> m_surrogate_loss;

		/** the helper that records primal objectives, duality gaps etc */
		std::shared_ptr<SOSVMHelper> m_helper;

		/** verbose outputs and statistics */
		bool m_verbose;

}; /* class StructuredOutputMachine */

} /* namespace shogun */

#endif /* _STRUCTURED_OUTPUT_MACHINE__H__ */
