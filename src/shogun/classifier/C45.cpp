#include <shogun/classifier/C45.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Labels.h>
#include <shogun/features/DotFeatures.h>


#include <stdio.h>

using namespace shogun;

static float64_t log_2=log(2.0);

const double BOUND = 1;
double Val[] = {  0,  0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00},
               Dev[] = {4.0,  3.09,  2.58,  2.33, 1.65, 1.28, 0.84, 0.25, 0.00};
double CF = 0.25;

double NUcfENdE(const double E, const double N)
{
    static double Coeff=0;
    double Val0, Pr, e, ans;

    if ( ! Coeff )
    {
        /*  Compute and retain the coefficient value, interpolating from

            the values in Val and Dev  */
        int i = 0;
        while ( CF > Val[i] ) i++;
        Coeff = Dev[i-1] + (Dev[i] - Dev[i-1]) * (CF - Val[i-1]) /(Val[i] - Val[i-1]);
        Coeff = Coeff * Coeff;
    }

    e = E;//E>N/2 ? N-E:E;
    if ( e < 1E-6 )
    {
        ans = N * (1 - exp(log(CF) / N));
    }
    else if ( e < 0.9999 )
    {
        Val0 = N * (1 - exp(log(CF) / N));
        ans = Val0 + e * (NUcfENdE(1.0, N) - Val0);
    }
    else if ( e + 0.5 >= N )
    {
        ans = 0.67 * (N - e);
    }
    else
    {
        Pr = (e + 0.5 + Coeff/2
              + sqrt(Coeff * ((e + 0.5) * (1 - (e + 0.5)/N) + Coeff/4)) )
             / (N + Coeff);
        ans = (N * Pr - e);
    }
    return ans;//E>N/2 ? N-ans:ans;
}

double UcfEN(const double E, const double N)
{
    static double Coeff=0;
    double Val0, Pr, e, ans;

    if (E==N) return 0;

    if ( ! Coeff )
    {
        /*  Compute and retain the coefficient value, interpolating from
            the values in Val and Dev  */
        int i = 0;
        while ( CF > Val[i] ) i++;
        Coeff = Dev[i-1] + (Dev[i] - Dev[i-1]) * (CF - Val[i-1]) /(Val[i] - Val[i-1]);
        Coeff = Coeff * Coeff;
    }

    e = E>N/2 ? N-E:E;
    if ( e < 1E-6 )
    {
        ans = (BOUND - exp(log(CF) / N));
    }
    else if ( e < 0.9999 )
    {
        Val0 = (BOUND - exp(log(CF) / N));
        ans = Val0 + e * (UcfEN(1.0, N) - Val0);
    }
    else if ( e + 0.5 >= N )
    {
        ans = 0.67 * (BOUND - e/N);
    }
    else
    {
        Pr = (e + 0.5 + Coeff/2
              + sqrt(Coeff * ((e + 0.5) * (1 - (e + 0.5)/N) + Coeff/4)) )
             / (N + Coeff);
        ans =  Pr;
    }
    return E>N/2 ? BOUND-ans:ans;
}

double NUcfEN(const double E, const double N)
{
    static double Coeff=0;
    double Val0, Pr, e, ans;

    if (E>=N) return 0;

    if ( ! Coeff )
    {
        /*  Compute and retain the coefficient value, interpolating from
            the values in Val and Dev  */
        int i = 0;
        while ( CF > Val[i] ) i++;
        Coeff = Dev[i-1] + (Dev[i] - Dev[i-1]) * (CF - Val[i-1]) /(Val[i] - Val[i-1]);
        Coeff = Coeff * Coeff;
    }

    e = E>N/2 ? N-E:E;
    if ( e < 1E-6 )
    {
        ans = N * (1 - exp(log(CF) / N));
    }
    else if ( e < 0.9999 )
    {
        Val0 = N * (1 - exp(log(CF) / N));
        ans = Val0 + e * (NUcfENdE(1.0, N) - Val0);
    }
    else if ( e + 0.5 >= N )
    {
        ans = 0.67 * (N - e);
    }
    else
    {
        Pr = (e + 0.5 + Coeff/2
              + sqrt(Coeff * ((e + 0.5) * (1 - (e + 0.5)/N) + Coeff/4)) )
             / (N + Coeff);
        ans = N * Pr;
    }
    return E>N/2 ? N-ans:ans;
}


CTreeNode::CTreeNode(int32_t num)
{
    this->attrib_id=-1;
    this->list=NULL;
    this->rightNode=NULL;
    this->leftNode=NULL;
    this->samples_count=num;
    if(num==0)
    {
        this->is_leaf=1;//is a leaf node when the samples set is empty
        this->index=NULL;
    }
    else
    {
        this->is_leaf=0;
        this->index=SG_MALLOC(int32_t,num);//the index array stores the indexes of samples included in this node
    }
}


bool CTreeNode::isLeaf()
{
    return LeafNode==this->is_leaf;
}

int32_t CTreeNode::getMajorClassLabel()
{
    return this->major_class;
}

int32_t CTreeNode::getMajorLabelCount()
{
    return this->samples_count - this->minor_count;
}

int32_t CTreeNode::getMinorLabelCount()
{
    return this->minor_count;
};

CTreeNode* CTreeNode::getNextChild()
{
    return this->rightNode;
}


CC45::CC45(CFeatures* t_examples, CLabels* t_labels) :
    CMachine(), m_features(NULL)
{
    ASSERT(t_examples->get_num_vectors()==t_labels->get_num_labels());
    set_labels(t_labels);
    set_features(t_examples);
    ASSERT(m_features);
    ASSERT(m_labels);
    //get number of feature vectors and dimensionality
    m_num_samples=m_features->get_num_vectors();
    m_dim=m_features->get_dim_feature_space();
};

CC45::~CC45()
{
    SG_UNREF(m_features);

    feature_matrix.destroy_matrix();
    train_labels.destroy_vector();
    is_discrete->destroy_vector();
    value_label->destroy_matrix();
}



bool CC45::train_machine(CFeatures* train_examples)
{
    // get int labels to train_labels and check length equality

    train_labels = m_labels->get_int_labels();

    ASSERT(m_features->get_num_vectors()==train_labels.vlen);

    feature_matrix = m_features->get_computed_dot_feature_matrix();

    yes_num = new SGVector<int32_t>(m_num_samples);
    no_num = new SGVector<int32_t>(m_num_samples);
    //initialize value_label
    value_label= new SGMatrix<float64_t>(m_num_samples,2);

    root = new CTreeNode(m_num_samples);
    root->minor_count=m_num_samples-1;
    for(int32_t i=0; i<m_num_samples; i++)
    {
        root->index[i]=i;
    }

    //iterative process
    generate_decision_tree(root);
    return true;
}

CTreeNode* CC45::generate_decision_tree(CTreeNode* n)
{
    /*************
    the first phase:
        return n as a leaf node,if samples included in n are all of the same class label
    ***************/
    current_tree_node=n;
    if(n->minor_count==0)
    {
        n->is_leaf=1;
        return n;
    }
    /*************
    the second phase:
    return n as a leaf node,if attribute list is empty or the count of samples included in n is less than MIN_SAMPLES_NUM
    ***************/
    if(attribute_list==NULL||n->samples_count<MIN_SAMPLES_NUM)
    {
        n->is_leaf=1;
        return n;
    }
    /*************
    ****************************
    the third phase
    ***********************
    ***************/
    int32_t split=-1;
    float64_t max=-10;
    float64_t tmp;
    SGVector<int32_t>* max_no_num =new SGVector<int32_t>(m_num_samples);
    SGVector<int32_t>* max_yes_num =new SGVector<int32_t>(m_num_samples);
    //initialize
    max_yes_num->zero();
    max_no_num->zero();

    CAttribNode* a=attribute_list;
    CAttribNode *max_a=a,*reserve;

    while(a!=NULL)
    {
        //reset
        yes_num->zero();
        no_num->zero();
        if(a->is_discrete==1)//a discretee-valued attribute
        {
            tmp=compute_gain_ratio_discreteized(a);
            if(tmp>max)
            {
                max=tmp;
                max_a=a;
                for(int i=0; i<m_num_samples; i++)
                {
                    (*max_yes_num)[i] = (*yes_num)[i];
                    (*max_no_num)[i] = (*no_num)[i];
                }

            }
        }
        else//a continuous-valued attribute
        {
            for(int i=0; i<current_tree_node->samples_count; i++)
            {
                (*value_label)[i*2+0]=feature_matrix.matrix[current_tree_node->index[i]*m_dim+a->id];
                (*value_label)[i*2+1]=train_labels[current_tree_node->index[i]];
            }
            quicksort(a->id,0,current_tree_node->samples_count-1);
            tmp=compute_gain_ratio_continuous(a,split=0);
            if (tmp>max)
            {
                max=tmp;
                max_a=a;
                for(int i=0; i<2; i++)
                {
                    (*max_yes_num)[i]=(*yes_num)[i];
                    (*max_no_num)[i]=(*no_num)[i];
                }
                n->threshold=(*value_label)[split*2+0];
            }
        }
        a=a->next;
    }

//delete the splitting attribute
    a=attribute_list;
    if(a==max_a)
    {
        attribute_list=a->next;
        reserve=NULL;
    }
    else
    {
        while(a->next!=max_a)
        {
            a=a->next;
        }
        reserve=a;
        a->next=a->next->next;
    }
    n->attrib_id=max_a->id;

    /*************
    ****************************
    the forth phase:
    split the samples included in the current node
    ***************/
    if(max_a->is_discrete==1)// discrete-valued attribute
    {
        CTreeNode *mid=NULL;
        CTreeNode *n_list=NULL;
        int j;//the index of child nodes of the current node
        for (int i=0; i<max_a->size; i++)
        {
            if (n_list==NULL)
            {
                n_list=new CTreeNode((*max_yes_num)[i]+(*max_no_num)[i]);
                mid=n_list;
            }
            else
            {
                n_list->list=new CTreeNode((*max_yes_num)[i]+(*max_no_num)[i]);
                n_list=n_list->list;
            }
            if(n_list->samples_count!=0)//the samples set isn't empty
            {
                if((*max_yes_num)[i]>(*max_no_num)[i])
                {
                    n_list->major_class=1;//represents the major class label is 1
                    n_list->minor_count=(*max_no_num)[i];
                }
                else
                {
                    n_list->major_class=0;//represents the major class label is 0
                    n_list->minor_count=(*max_yes_num)[i];
                }
                j=0;
                for (int k=0; k<n->samples_count; k++)
                {
                    if(feature_matrix.matrix[n->index[k]*m_dim+max_a->id]==(float64_t)i)
                    {
                        n_list->index[j]=n->index[k];

                        j++;
                    }
                }
                n_list->samples_count=j;
                generate_decision_tree(n_list);
            }
            else
            {
                n_list->major_class=n->major_class; //assign major_class of father node to major_class of child node
            }
        }
        n->rightNode=mid;
    }
    else //continous-valued attribute
    {
        //right child node is not a leaf node by default
        CTreeNode* n_right=new CTreeNode((*max_yes_num)[1]+(*max_no_num)[1]+1);
        //left child node is not a leaf node by default
        CTreeNode* n_left=new CTreeNode((*max_yes_num)[0]+(*max_no_num)[0]+1);

        if((*max_yes_num)[0]>(*max_no_num)[0])
        {
            n_left->major_class=1;
            n_left->minor_count=(*max_no_num)[0];
        }
        else
        {
            n_left->major_class=0;
            n_left->minor_count=(*max_yes_num)[0];
        }
        if((*max_yes_num)[1]>(*max_no_num)[1])
        {
            n_right->major_class=1;
            n_right->minor_count=(*max_no_num)[1];
        }
        else
        {
            n_right->major_class=0;
            n_right->minor_count=(*max_yes_num)[1];
        }
        int &j=n_left->samples_count,&k=n_right->samples_count,i;
        for (i=0,j=0,k=0; i<n->samples_count; i++)
        {
            if(feature_matrix.matrix[n->index[i]*m_dim+max_a->id]<=n->threshold)
            {
                n_left->index[j]=n->index[i];
                j++;
            }
            else
            {
                n_right->index[k]=n->index[i];
                k++;
            }
        }
        n->leftNode=generate_decision_tree(n_left);
        n->rightNode=generate_decision_tree(n_right);
    }

    //add the splitting attribute removed to the attribute list
    if(reserve==NULL)
    {
        max_a->next=attribute_list;
        attribute_list=max_a;
    }
    else
    {
        max_a->next=reserve->next;
        reserve->next=max_a;
    }
    return n;
}

// compute the gain ratio when splitting samples by a discrete-valued attribute
float64_t CC45::compute_gain_ratio_discreteized(CAttribNode* a)
{
    float64_t gain_ratio=0;
    float64_t gain=info;
    float64_t split_info=0;
    float64_t k1,k2,t2;
    int32_t i,j,t1;
    int32_t *p=current_tree_node->index,*q=p+current_tree_node->samples_count;
    while(p<q)
    {
        t1=(*p)*m_dim;
        i=(int32_t)feature_matrix.matrix[t1+a->id];
        if(train_labels[*p]==1)
        {
            (*yes_num)[i]++;
        }
        else
        {
            (*no_num)[i]++;
        }
        p++;
    }
    for (i=0; i<a->size; i++)
    {
        j=(*yes_num)[i]+(*no_num)[i];
        if(j!=0)
        {
            if ((*yes_num)[i]==0)
            {
                k1=0;
            }
            else
            {
                t2=(*yes_num)[i]/(float64_t)j;
                k1=t2*log(t2)/log_2;
            }
            if ((*no_num)[i]==0)
            {
                k2=0;
            }
            else
            {
                t2=(*no_num)[i]/(float64_t)j;
                k2=t2*log(t2)/log_2;
            }
            t2=j/(float64_t)current_tree_node->samples_count;
            gain+=((k1+k2)*t2);
            split_info-=(t2*log(t2)/log_2);
        }
    }
    gain_ratio=split_info==0.0?MIN_GAIN_RATIO:gain/split_info;
    return gain_ratio;
}


// compute the gain ratio when splitting samples by a continuous-valued attribute
float64_t CC45::compute_gain_ratio_continuous(CAttribNode* a, int32_t &max_split)
{
    float64_t gain_ratio=0,max_gain_ratio=MIN_GAIN_RATIO;
    float64_t split_info;
    float64_t gain;
    int32_t i,j,t,split,max=current_tree_node->samples_count-1,max2=max-1;
    float64_t k1,k2,t2;//temp
    for (i=0; i<max; i++)
    {
        split=(*value_label)[i*2+0];
        while((*value_label)[(i+1)*2+0]==split&&i<max2)
        {
            i++;
        }
        gain=info;
        split_info=0;
        (*yes_num)[2]=0;
        (*no_num)[2]=0;
        (*yes_num)[3]=0;
        (*no_num)[3]=0;

        for(split=0; split<=i; split++)
        {
            //leftNode
            if((*value_label)[split*2+1]==1)//initialized in sorting
            {
                (*yes_num)[2]++;
            }
            else
            {
                (*no_num)[2]++;
            }
        }
        split--;
        for(j=split+1; j<current_tree_node->samples_count; j++)
        {
            //rightNode
            if((*value_label)[j*2+1]==1)
            {
                (*yes_num)[3]++;
            }
            else
            {
                (*no_num)[3]++;
            }
        }
        for (t=2; t<4; t++)
        {
            j=(*yes_num)[t]+(*no_num)[t];
            if (j==0)//
            {
                continue;
            }
            if ((*yes_num)[t]==0)
            {
                k1=0;
            }
            else
            {
                t2=(*yes_num)[t]/(float64_t)j;
                k1=t2*log(t2)/log_2;
            }
            if ((*no_num)[t]==0)
            {
                k2=0;
            }
            else
            {
                t2=(*no_num)[t]/(float64_t)j;
                k2=t2*log(t2)/log_2;
            }
            t2=j/(float64_t)current_tree_node->samples_count;
            gain+=((k1+k2)*t2);
            split_info-=(t2*log(t2)/log_2);
        }
        gain_ratio=gain/split_info;
        if (gain_ratio>max_gain_ratio)
        {
            max_gain_ratio=gain_ratio;
            max_split=split;
            (*yes_num)[0]=(*yes_num)[2];
            (*no_num)[0]=(*no_num)[2];
            (*yes_num)[1]=(*yes_num)[3];
            (*no_num)[1]=(*no_num)[3];
        }
    }
    return max_gain_ratio;
}

float64_t CC45::compute_error_ratio(CTreeNode* n)
{
    float64_t e,N;
    N = (float64_t)n->samples_count;
    e = (float64_t)n->minor_count;
    n->ucfErrorsRate=NUcfEN(e,N);

    if (n->is_leaf)
    {
        n->error_rate = n->expErrorsRate = n->ucfErrorsRate;
        //cout << "A leaf.\tN:" << N << "\te:" << e << endl;
        //cout << "\terror_rate=" << n->error_rate << "\tucfErrorsRate(e:"<<e<< ",N:"<<N<<")=" << n->ucfErrorsRate << "\texpErrorsRate=" << n->expErrorsRate << endl;
        return n->error_rate;
    }
    if (NULL==n->leftNode) //按照离散属性划分的
    {
        CTreeNode* cl = n->rightNode;
        n->expErrorsRate = 0;
        do
        {
            n->expErrorsRate+=compute_error_ratio(cl);
            cl=cl->list;
        }
        while(cl!=NULL);
    }
    else
    {
        n->expErrorsRate=compute_error_ratio(n->rightNode)+compute_error_ratio(n->leftNode);
    }

    if (n->ucfErrorsRate < n->expErrorsRate)
    {
        n->error_rate = n->ucfErrorsRate;
        n->is_leaf = CutLeafNode;
        //cout << "A Cutleaf.";
    }
    else
    {
        n->error_rate = n->expErrorsRate;
        //cout << "A Banch.";
    }
    //cout << "\tN:" << N << "\te:" << e << endl;
    //cout << "\terror_rate=" << n->error_rate << "\tucfErrorsRate(e:"<<e<< ",N:"<<N<<")=" << n->ucfErrorsRate << "\texpErrorsRate=" << n->expErrorsRate << endl;
    return n->error_rate;
}

//idx represents the id of attribute
int32_t CC45::partition(int32_t idx, int32_t left, int32_t right)
{
    int low,high,mid,t;
    low=left+1;
    high=right;
    mid=(*value_label)[left*2+0];
    while (low<=high)
    {
        if ((*value_label)[low*2+0]<=mid)
        {
            low++;
        }
        else if((*value_label)[high*2+0]>mid)
        {
            high--;
        }
        else
        {
            t=(*value_label)[low*2+0];
            (*value_label)[low*2+0]=(*value_label)[high*2+0];
            (*value_label)[high*2+0]=t;
            //
            t=(*value_label)[low*2+1];
            (*value_label)[low*2+1]=(*value_label)[high*2+1];   //Yes/No
            (*value_label)[high*2+1]=t;
            low++;
            high--;
        }
    }
    (*value_label)[left*2+0]=(*value_label)[high*2+0];
    (*value_label)[high*2+0]=mid;
    t=(*value_label)[left*2+1];
    (*value_label)[left*2+1]=(*value_label)[high*2+1];
    (*value_label)[high*2+1]=t;
    return high;
}

//sort the values of a continuous attribute with the attribute idx
void CC45::quicksort(int32_t idx, int32_t left, int32_t right)
{
    int32_t mid;
    if(left<right)
    {
        mid=partition(idx,left,right);
        quicksort(idx,left,mid-1);//recursive procedure,left side
        quicksort(idx,mid+1,right);//recursive procedure,right side
    }
}

void CC45::set_attribute_list(SGMatrix<int32_t> values)
{
    is_discrete=new SGVector<int32_t>(m_dim);
    attribute_list=new CAttribNode;
    CAttribNode *iterator=attribute_list;
    int i;
    int col=values.num_cols;
    i=0;
    while(1)
    {
        iterator->id=i;
        if(values[i*col+0]==0)
        {
            iterator->is_discrete=0;
            (*is_discrete)[i]=0;
        }
        else
        {
            iterator->is_discrete=1;
            (*is_discrete)[i]=1;
            iterator->size=values[i*col+1];
        }
//		if(values[i*col+0]==-1)
//		{
//			iterator->is_discrete=0;
//			(*is_discrete)[i]=0;
//		}
//		else{
//			iterator->is_discrete=1;
//			(*is_discrete)[i]=1;
//			j=0;
//			iterator->value=SG_MALLOC(int32_t,values.num_cols);
//			while(values[i*col+j]!=-1)
//			{
//				iterator->value[j]=values[i*col+j];
//				j++;
//			}
//			iterator->size=j;
//		}
        if(++i>=m_dim)
            break;
        iterator->next=new CAttribNode;
        iterator=iterator->next;
    }
    iterator->next=NULL;
}

void  CC45::set_info(float64_t information)
{
    info=information;
}


CLabels* CC45::apply()
{
    // init number of vectors
    int32_t n = m_features->get_num_vectors();

    // init result labels
    CLabels* result = new CLabels(n);

    // classify each example of data
    for (int i=0; i<n; i++)
        result->set_label(i,apply(i));

    return result;
}

CLabels* CC45::apply(CFeatures* data)
{
    // check data correctness
    if (!data)
        SG_ERROR("No features specified\n");

    // set features to classify
    set_features(data);

    // classify using features
    return apply();
}

float64_t CC45::apply(int32_t idx)
{
    float64_t result_label=dfs(root,idx);
    return result_label;
}

float64_t CC45::dfs(CTreeNode* r,int k)
{
    // get [k] feature vector
    int value=(int)feature_matrix.matrix[k*m_dim+r->attrib_id];
    int tmp;
    if (r==NULL)
    {
        return -2;
    }
    if (r->is_leaf)
    {
        return r->major_class;
    }
    if (r->leftNode==NULL)//represents a discrete-valued attribute
    {
        CTreeNode* child=r->rightNode;
        for(tmp=0; tmp<value; tmp++)
        {
            child=child->list;
        }
        if (child!=NULL)
        {
            return dfs(child,k);
        }
        else
        {
            return -1;
        }
    }
    else
    {
        //a continuous-valued attribute
        tmp=value;
        if (tmp<=r->threshold)
        {
            return dfs(r->leftNode,k);
        }
        else
        {
            return dfs(r->rightNode,k);
        }
    }
}


