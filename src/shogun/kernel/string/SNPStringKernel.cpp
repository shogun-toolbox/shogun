/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Fernando Iglesias,
 *          Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/SNPStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

SNPStringKernel::SNPStringKernel()
: StringKernel<char>(0),
  m_degree(0), m_win_len(0), m_inhomogene(false)
{
	init();
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	register_params();
}

SNPStringKernel::SNPStringKernel(int32_t size,
		int32_t degree, int32_t win_len, bool inhomogene)
: StringKernel<char>(size),
	m_degree(degree), m_win_len(2*win_len), m_inhomogene(inhomogene)
{
	init();
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	register_params();
}

SNPStringKernel::SNPStringKernel(
	const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
	int32_t degree, int32_t win_len, bool inhomogene)
: StringKernel<char>(10), m_degree(degree), m_win_len(2*win_len),
	m_inhomogene(inhomogene)
{
	init();
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	if (l==r)
		obtain_base_strings();
	init(l, r);
	register_params();
}

SNPStringKernel::~SNPStringKernel()
{
	cleanup();
}

bool SNPStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void SNPStringKernel::cleanup()
{
	Kernel::cleanup();
	SG_FREE(m_str_min);
	SG_FREE(m_str_maj);
}

void SNPStringKernel::obtain_base_strings()
{
	//should only be called on training data
	ASSERT(lhs==rhs)

	m_str_len=0;

	auto sf = std::static_pointer_cast<StringFeatures<char>>(lhs);
	for (int32_t i=0; i<num_lhs; i++)
	{
		int32_t len;
		bool free_vec;
		char* vec = sf->get_feature_vector(i, len, free_vec);

		if (m_str_len==0)
		{
			m_str_len=len;
			m_str_min=SG_CALLOC(char, len+1);
			m_str_maj=SG_CALLOC(char, len+1);
		}
		else
		{
			ASSERT(m_str_len==len)
		}

		for (int32_t j=0; j<len; j++)
		{
			// skip sequencing errors
			if (vec[j]=='0')
				continue;

			if (m_str_min[j]==0)
				m_str_min[j]=vec[j];
            else if (m_str_maj[j]==0 && vec[j]!=m_str_min[j])
				m_str_maj[j]=vec[j];
		}

		sf->free_feature_vector(vec, i, free_vec);
	}

	for (int32_t j=0; j<m_str_len; j++)
	{
        // if only one one symbol occurs use 0
		if (m_str_min[j]==0)
            m_str_min[j]='0';
		if (m_str_maj[j]==0)
            m_str_maj[j]='0';

		if (m_str_min[j]>m_str_maj[j])
			Math::swap(m_str_min[j], m_str_maj[j]);
	}
}

float64_t SNPStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)
	if (alen!=m_str_len)
		error("alen ({}) !=m_str_len ({})", alen, m_str_len);
	ASSERT(m_str_min)
	ASSERT(m_str_maj)

	float64_t total=0;
	int32_t inhomogene= (m_inhomogene) ? 1 : 0;

	for (int32_t i = 0; i<alen-1; i+=2)
	{
		int32_t sumaa=0;
		int32_t sumbb=0;
		int32_t sumab=0;

		for (int32_t l=0; l<m_win_len && i+l<alen-1; l+=2)
		{
			char a1=avec[i+l];
			char a2=avec[i+l+1];
			char b1=bvec[i+l];
			char b2=bvec[i+l+1];

			if ((a1!=a2 || a1=='0' || a2=='0') && (b1!=b2 || b1=='0' || b2=='0'))
				sumab++;
			else if (a1==a2 && b1==b2)
			{
				if (a1!=b1)
					continue;

				if (a1==m_str_min[i+l])
					sumaa++;
				else if (a1==m_str_maj[i+l])
					sumbb++;
				else
				{
					error("The impossible happened i={} l={} a1={} "
							"a2={} b1={} b2={} min={} maj={}", i, l, a1,a2, b1,b2, m_str_min[i+l], m_str_maj[i+l]);
				}
			}

		}
		total+=Math::pow(float64_t(sumaa+sumbb+sumab+inhomogene),
				(int32_t) m_degree);
	}

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return total;
}

void SNPStringKernel::register_params()
{
	SG_ADD(&m_degree, "m_degree", "the order of the kernel", ParameterProperties::HYPER);
	SG_ADD(&m_win_len, "m_win_len", "the window length", ParameterProperties::HYPER);
	SG_ADD(&m_inhomogene, "m_inhomogene",
	  "the mark of whether it's an inhomogeneous poly kernel");

	/*m_parameters->add_vector(&m_str_min, &m_str_len, "m_str_min", "allele A");*/
	watch_param("m_str_min", &m_str_min, &m_str_len);

	/*m_parameters->add_vector(&m_str_maj, &m_str_len, "m_str_maj", "allele B");*/
	watch_param("m_str_maj", &m_str_maj, &m_str_len);
}

void SNPStringKernel::init()
{
	m_str_min=NULL;
	m_str_maj=NULL;
	m_str_len=0;
}
