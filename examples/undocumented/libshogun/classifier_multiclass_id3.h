
using namespace std;

namespace shogun{

struct id3_node
{		
		int attribute;
		int attribute_label;
		std::vector<id3_node> children;
		id3_node* parent;
};

class CID3Classifier
{
public:
	/** constructor */
	CID3Classifier();

	/** destructor */
	virtual ~CID3Classifier();

	/** get name */
	virtual const char* get_name() const { return "ID3Classifier"; }
	
	/** set labels
	 *
	 * @param lab labels
	 */
	void set_labels(CLabels* lab);
	
	/** set features
	 * @param feats features
	 */
	void set_features(CDenseFeatures<float64_t> *feats);		
	/** set names
	 * @param names of the attributes, names of the labels for each attribute
	 */	
	void set_names(std::vector<string> v1,	std::vector< std::vector<string> > v2)
	{
		int i;
		attribute_names.insert(attribute_names.end(), v1.begin(), v1.end());

		names = v2;
	
	}

	/** train 
	 *
	 * @param data training data
	 * @return the root of the ID3 tree
	 */
	id3_node train(CFeatures* data, CMulticlassLabels *class_labels, SGVector<int32_t> *values = NULL, int level = 0);
	
	/** evaluate
	 *
	 * @param sample to test and the root of the ID3 node
	 * @return the class the sample belongs to
	 */	
	int evaluate(SGVector<float64_t> sample, id3_node node);
	
	void print_dataset(CFeatures* data, CMulticlassLabels *class_labels);
	void print_id3_tree(id3_node* root, int tab = 0);	
	
	/** informational_gain_attribute
	 *
	 * @param attribute id, data training data, classes each sample in the training set belongs to 
	 * @return informational gain
	 */	
	float64_t informational_gain_attribute(int32_t attr_no, CFeatures* data, CMulticlassLabels *class_labels);	
	
	/** informational_gain_attribute
	 *
	 * @param a set of lables for an attribute
	 * @return entropy
	 */		
	float64_t entropy(CMulticlassLabels *labels);	

	/** features */
	CDenseFeatures<float64_t> *m_feats;
	
	/** attributes index vector **/
	SGVector<int32_t> *attributes;	
	
	/** class labels */
	CMulticlassLabels *class_labels;
		
	/** lable names **/
	std::vector<string> attribute_names;
	std::vector< std::vector<string> > names;	

	/** number of classes */
	int32_t m_num_classes;
	
};

}
