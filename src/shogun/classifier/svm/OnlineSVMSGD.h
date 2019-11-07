#ifndef _ONLINESVMSGD_H___
#define _ONLINESVMSGD_H___
/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Shashwat Lal Das, Sergey Lisitsyn, 
 *          Fernando Iglesias, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/OnlineLinearMachine.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief class OnlineSVMSGD */
class OnlineSVMSGD : public OnlineLinearMachine
{
	public:
		/** returns type of problem machine solves */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor  */
		OnlineSVMSGD();

		/** constructor
		 *
		 * @param C constant C
		 */
		OnlineSVMSGD(float64_t C);

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 */
		OnlineSVMSGD(float64_t C, std::shared_ptr<StreamingDotFeatures> traindat);

		virtual ~OnlineSVMSGD();

		/** get classifier type
		 *
		 * @return classifier type OnlineSVMSGD
		 */
		virtual EMachineType get_classifier_type() { return CT_SVMSGD; }

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(std::shared_ptr<Features> data=NULL);

		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set epochs
		 *
		 * @param e new number of training epochs
		 */
		inline void set_epochs(int32_t e) { epochs=e; }

		/** get epochs
		 *
		 * @return the number of training epochs
		 */
		inline int32_t get_epochs() { return epochs; }

		/** set lambda
		 *
		 * @param l value of regularization parameter lambda
		 */
		inline void set_lambda(float64_t l) { lambda=l; }

		/** get lambda
		 *
		 * @return the regularization parameter lambda
		 */
		inline float64_t get_lambda() { return lambda; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set if regularized bias shall be enabled
		 *
		 * @param enable_bias if regularized bias shall be enabled
		 */
		inline void set_regularized_bias_enabled(bool enable_bias) { use_regularized_bias=enable_bias; }

		/** check if regularized bias is enabled
		 *
		 * @return if regularized bias is enabled
		 */
		inline bool get_regularized_bias_enabled() { return use_regularized_bias; }

		/** Set the loss function to use
		 *
		 * @param loss_func object derived from CLossFunction
		 */
		void set_loss_function(std::shared_ptr<LossFunction> loss_func);

		/** Return the loss function
		 *
		 * @return loss function as CLossFunction*
		 */
		inline std::shared_ptr<LossFunction> get_loss_function() {  return loss; }

		/** @return object name */
		inline const char* get_name() const { return "OnlineSVMSGD"; }

	protected:
		/** calibrate
		 *
		 * @param max_vec_num Maximum number of vectors to calibrate using
		 * (optional) if set to -1, tries to calibrate using all vectors
		 * */
		void calibrate(int32_t max_vec_num=1000);

	private:
		void init();

	private:
		float64_t t;
		float64_t lambda;
		float64_t C1;
		float64_t C2;
		float64_t wscale;
		float64_t bscale;
		int32_t epochs;
		int32_t skip;
		int32_t count;

		bool use_bias;
		bool use_regularized_bias;

		std::shared_ptr<LossFunction> loss;
};
}
#endif
