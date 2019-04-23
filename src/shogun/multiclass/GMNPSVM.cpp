/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang, Heiko Strathmann, Sergey Lisitsyn
 */

#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/GMNPSVM.h>
#include <shogun/multiclass/GMNPLib.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW))
#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX
#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

using namespace shogun;

GMNPSVM::GMNPSVM()
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>())
{
	init();
}

GMNPSVM::GMNPSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>(), C, k, lab)
{
	init();
}

GMNPSVM::~GMNPSVM()
{
	if (m_basealphas != NULL) SG_FREE(m_basealphas);
}

void
GMNPSVM::init()
{
	/*m_parameters->add_matrix(&m_basealphas,
							 &m_basealphas_y, &m_basealphas_x,
							 "m_basealphas",
							 "Is the basic untransformed alpha.");*/
	watch_param(
	    "m_basealphas", &m_basealphas, &m_basealphas_y, &m_basealphas_x);

	m_basealphas = NULL, m_basealphas_y = 0, m_basealphas_x = 0;
}

bool GMNPSVM::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(m_kernel)
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
	init_strategy();

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
		{
			SG_ERROR("%s::train_machine(): Number of training vectors (%d) does"
					" not match number of labels (%d)\n", get_name(),
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		m_kernel->init(data, data);
	}

	int32_t num_data = m_labels->get_num_labels();
	int32_t num_classes = m_multiclass_strategy->get_num_classes();
	int32_t num_virtual_data= num_data*(num_classes-1);

	SG_INFO("%d trainlabels, %d classes\n", num_data, num_classes)

	float64_t* vector_y = SG_MALLOC(float64_t, num_data);
	auto mc = multiclass_labels(m_labels);
	for (int32_t i=0; i<num_data; i++)
	{
		vector_y[i] = mc->get_label(i)+1;

	}

	float64_t C = get_C();
	int32_t tmax = 1000000000;
	float64_t tolabs = 0;
	float64_t tolrel = get_epsilon();

	float64_t reg_const=0;
	if( C!=0 )
		reg_const = 1/(2*C);


	float64_t* alpha = SG_MALLOC(float64_t, num_virtual_data);
	float64_t* vector_c = SG_MALLOC(float64_t, num_virtual_data);
	memset(vector_c, 0, num_virtual_data*sizeof(float64_t));

	float64_t thlb = 10000000000.0;
	int32_t t = 0;
	float64_t* History = NULL;
	int32_t verb = 0;

	GMNPLib mnp(vector_y,m_kernel,num_data, num_virtual_data, num_classes, reg_const);

	mnp.gmnp_imdm(vector_c, num_virtual_data, tmax,
				  tolabs, tolrel, thlb, alpha, &t, &History, verb);

	/* matrix alpha [num_classes x num_data] */
	float64_t* all_alphas= SG_MALLOC(float64_t, num_classes*num_data);
	memset(all_alphas,0,num_classes*num_data*sizeof(float64_t));

	/* bias vector b [num_classes x 1] */
	float64_t* all_bs=SG_MALLOC(float64_t, num_classes);
	memset(all_bs,0,num_classes*sizeof(float64_t));

	/* compute alpha/b from virt_data */
	for(int32_t i=0; i < num_classes; i++ )
	{
		for(int32_t j=0; j < num_virtual_data; j++ )
		{
			int32_t inx1=0;
			int32_t inx2=0;

			mnp.get_indices2( &inx1, &inx2, j );

			all_alphas[(inx1*num_classes)+i] +=
				alpha[j]*(KDELTA(vector_y[inx1],i+1)-KDELTA(i+1,inx2));
			all_bs[i] += alpha[j]*(KDELTA(vector_y[inx1],i+1)-KDELTA(i+1,inx2));
		}
	}

	create_multiclass_svm(num_classes);

	for (int32_t i=0; i<num_classes; i++)
	{
		int32_t num_sv=0;
		for (int32_t j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
				num_sv++;
		}
		ASSERT(num_sv>0)
		SG_DEBUG("svm[%d] has %d sv, b=%f\n", i, num_sv, all_bs[i])

		auto svm=std::make_shared<SVM>(num_sv);

		int32_t k=0;
		for (int32_t j=0; j<num_data; j++)
		{
			if (all_alphas[j*num_classes+i] != 0)
			{
				svm->set_alpha(k, all_alphas[j*num_classes+i]);
				svm->set_support_vector(k, j);
				k++;
			}
		}

		svm->set_bias(all_bs[i]);
		set_svm(i, svm);
	}

	if (m_basealphas != NULL) SG_FREE(m_basealphas);
	m_basealphas_y = num_classes, m_basealphas_x = num_data;
	m_basealphas = SG_MALLOC(float64_t, m_basealphas_y*m_basealphas_x);
	for (index_t i=0; i<m_basealphas_y*m_basealphas_x; i++)
		m_basealphas[i] = 0.0;

	for(index_t j=0; j<num_virtual_data; j++)
	{
		index_t inx1=0, inx2=0;

		mnp.get_indices2(&inx1, &inx2, j);
		m_basealphas[inx1*m_basealphas_y + (inx2-1)] = alpha[j];
	}

	SG_FREE(vector_c);
	SG_FREE(alpha);
	SG_FREE(all_alphas);
	SG_FREE(all_bs);
	SG_FREE(vector_y);
	SG_FREE(History);

	return true;
}

float64_t*
GMNPSVM::get_basealphas_ptr(index_t* y, index_t* x)
{
	if (y == NULL || x == NULL) return NULL;

	*y = m_basealphas_y, *x = m_basealphas_x;
	return m_basealphas;
}
