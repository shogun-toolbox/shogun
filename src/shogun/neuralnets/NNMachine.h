#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
class CBinaryLabels;
class CMulticlassLabels;
class CFeatures;
class SGVector;

class CNNMachine : public CMachine
{
	public:
		/** constructor */
		CNNMachine();

		/** destructor */
		virtual ~CNNMachine();

		/** get weights at layer n
		 *
		 * @return weight vector
		 */
		virtual SGVector<float64_t> get_w(int32_t layer_n) const;

		/** set wegihts at layer n
		 *
		 * @param src_w new w
		 */
		virtual void set_w(const SGVector<float64_t>, int32_t layer_n);

		/** set bias at layer n
		 *
		 * @param b new bias
		 */
		virtual void set_bias(float64_t b, int32_t layer_n);

		/** get bias at layer n
		 *
		 * @return bias
		 */
		virtual float64_t get_bias(int32_t layer_n);

		/** apply NN machine to data
		 * for binary classification problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

		/** apply NN machine to data
		 * for multi-class classification problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CMulticlassLabels apply_multiclass(CFeatures* data=NULL);

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "NNMachine"; }

	private:
		/** register parameters and init NN machine */
		void init();
};
}
