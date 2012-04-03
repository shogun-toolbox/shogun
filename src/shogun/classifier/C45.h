#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DotFeatures.h>



namespace shogun
{


//the recursive procedure of generating decision is stopped when there're MIN_SAMPLES_NUM samples left
#define MIN_SAMPLES_NUM 2

#define MIN_GAIN_RATIO -10

//define the types of a tree node
enum NodeType {OtherNode, LeafNode, CutLeafNode};

//attribute node
class CAttribNode
{
public:
    int32_t id;//the index of a attribute node in the attribute list
    int32_t is_discrete;//is discrete?
    int32_t *value;//a array of the values of a  attribute node
    int32_t size;//the size of value array
    CAttribNode *next;//next node in attribute list
};

//tree node
class CTreeNode
{
public:
    CTreeNode(int32_t num);
    bool isLeaf();
    int32_t getMajorLabelCount();
    int32_t getMinorLabelCount();
    int32_t getMajorClassLabel();
    CTreeNode *getNextChild();
public:
    CTreeNode *list;        //the brother list
    CTreeNode *leftNode;       //left child
    CTreeNode *rightNode;      //right child
    int32_t attrib_id;         //the splitting attribute
    int32_t threshold;            //the splitting threshold
    int32_t *index;          //store the index of samples included in this tree node
    int32_t samples_count;     //the size of index array
    float64_t error_rate;
    float64_t ucfErrorsRate;  //the upper limit of error rate as a leaf
    float64_t expErrorsRate;	//the error rate not as a leaf
    int32_t is_leaf;          //1 represents leaf node, 0 represents other nodes
    int32_t major_class;//the major class label of the samples included in this tree node
    int32_t minor_count;//the count of samples with minor class label
};


class CDotFeatures;
class CMachine;

class CC45 : public CMachine
{
public:
    CC45():CMachine(), m_features(NULL) {};

    CC45(CFeatures *t_examples, CLabels *t_labels);

    ~CC45();

    inline void set_features(CFeatures *features)
    {
        SG_UNREF(m_features);
        SG_REF(features);
        m_features = (CDotFeatures*)features;
    }

    inline CFeatures *get_features()
    {
        SG_REF(m_features);
        return m_features;
    }

    inline const char *get_name() const
    {
        return "C45";
    };

    void  set_info(float64_t information);

    void set_attribute_list(SGMatrix<int32_t> values);

    bool train_machine(CFeatures *train_examples = NULL);

    CLabels *apply();

    CLabels *apply(CFeatures *data);

    float64_t apply(int32_t idx);

protected:
    CTreeNode *generate_decision_tree(CTreeNode *n);

    float64_t compute_gain_ratio_discreteized(CAttribNode *a);

    float64_t compute_gain_ratio_continuous(CAttribNode *a, int32_t &max_split);

    float64_t compute_error_ratio(CTreeNode *n);

    CAttribNode *get_attribute_list();

    float64_t dfs(CTreeNode *r,int k);

    void quicksort(int32_t idx, int32_t left, int32_t right);

    float64_t get_info();

private:
    int32_t partition(int32_t idx, int32_t left, int32_t right);

protected:
    int32_t m_num_samples;//the number of training samples

    int32_t m_dim;//the dimension of feature/attribute

    float64_t info;//the amount of information

    CTreeNode *root;

    CTreeNode *current_tree_node;

    CDotFeatures *m_features;

    SGMatrix<float64_t> feature_matrix;

    SGMatrix<float64_t> *value_label;

    SGVector<int32_t> train_labels;

    CAttribNode *attribute_list;//the list of attributes

    SGVector<int32_t> *is_discrete;

    SGVector<int32_t> *yes_num;

    SGVector<int32_t> *no_num;
};
}
