#include <shogun/lib/common.h>
#include <shogun/neuralnets/NNMachine.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
class CBinaryLabels;
class CMulticlassLabels;
class CFeatures;
class SGMatrix;
class SGVector;

class CBPNNMachine : public CNNMachine
{
	public:
		/** constructor */
		CBPNNMachine();

		/** destructor */
		virtual ~CBPNNMachine();

		/** train neural network
		 *
		 * @param data training data
		 * @return whether training was successful
		 */
		virtual bool train_all(CFeatures* data=NULL);

		/** test neural network
		 *
		 * @param data testing data
		 * @return whether testing was successful
		 */
		virtual bool test_all(CFeatures* data=NULL);

		/** train neural network with minibatch data
		 *
		 * @param data minibatch training data
		 * @return whether training was successful
		 */
		virtual bool train_minibatch(CFeatures* data=NULL);

		/** test neural network with minibatch data
		 *
		 * @param data minibatch testing data
		 * @return whether testing was successful
		 */
		virtual bool test_minibatch(CFeatures* data=NULL);

	protected:
		/** init an epoch
		 *
		 * @param cur_epoch current epoch
		 * @return whether init successfully
		 */
		virtual bool init_epoch(int32_t cur_epoch);

		/** feedforward
		 *
		 * @param inputs input features
		 * @param true_outputs true labels
		 * @return error calulated by NN outputs and true labels
		 */
		virtual float32_t feedforward(SGMatrix &inputs, const SGMatrix &true_output);

		/** backpropagation (calculate gradients)
		 *
		 * @param inputs
		 */
		virtual void backpropagation(SGMatrix &inputs);

		/** apply gradients */
		virtual void apply_gradients();

		/** true labels */
		SGMatrix true_output;
		/** weights among all layers */
		SGMatrix* m_weights;
		/** bias among all layers */
		SGVector* m_bias;
		/** derivatives of weights */
		SGMatrix* m_dw;
		/** intermediate variable to calculate weight momentum */
		SGMatrix* m_vw;
		/** derivatives of bias */
		SGVector* m_db;
		/** intermediate variable to calculate bias momentum */
		SGVector* m_vb;
		/** error in one layer */
		SGMatrix m_err;
		/** activations */
		SGMatrix* m_activations;
};
}
