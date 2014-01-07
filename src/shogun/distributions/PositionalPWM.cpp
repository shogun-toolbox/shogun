/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <distributions/PositionalPWM.h>
#include <mathematics/Math.h>
#include <base/Parameter.h>
#include <features/Alphabet.h>
#include <features/StringFeatures.h>

using namespace shogun;

CPositionalPWM::CPositionalPWM() : CDistribution(),
	m_sigma(0), m_mean(0)
{
	m_pwm = SGMatrix<float64_t>();
	m_w = SGMatrix<float64_t>();
	m_poim = SGVector<float64_t>();

	register_params();
}

CPositionalPWM::~CPositionalPWM()
{
}

bool CPositionalPWM::train(CFeatures* data)
{
	SG_NOTIMPLEMENTED
	return true;
}

int32_t CPositionalPWM::get_num_model_parameters()
{
	return m_pwm.num_rows*m_pwm.num_cols+2;
}

float64_t CPositionalPWM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param>0 && num_param<=m_pwm.num_rows*m_pwm.num_cols+2)

	if (num_param<m_pwm.num_rows*m_pwm.num_cols)
	{
		return m_pwm[num_param];
	}
	else if (num_param<m_pwm.num_rows*m_pwm.num_cols+1)
		return CMath::log(m_sigma);
	else
		return CMath::log(m_mean);
}

float64_t CPositionalPWM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED
	return 0;
}

float64_t CPositionalPWM::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features)
	ASSERT(features->get_feature_class() == C_STRING)
	ASSERT(features->get_feature_type()==F_BYTE)

	CStringFeatures<uint8_t>* strs=(CStringFeatures<uint8_t>*) features;

	float64_t lik=0;
	int32_t len=0;
	bool do_free=false;

	uint8_t* str = strs->get_feature_vector(num_example, len, do_free);

	if (!(m_w.num_cols==len))
		return 0; //TODO

	for (int32_t i=0; i<len; i++)
		lik+=m_w[4*i+str[i]];

	strs->free_feature_vector(str, num_example, do_free);
	return lik;
}

float64_t CPositionalPWM::get_log_likelihood_window(uint8_t* window, int32_t len, float64_t pos)
{
	ASSERT(m_pwm.num_cols == len)
	float64_t score = CMath::log(1/(m_sigma*CMath::sqrt(2*M_PI))) -
			CMath::sq(pos-m_mean)/(2*CMath::sq(m_sigma));

	for (int32_t i=0; i<m_pwm.num_cols; i++)
		score+=m_pwm[m_pwm.num_rows*i+window[i]];

	return score;
}

void CPositionalPWM::compute_w(int32_t num_pos)
{
	ASSERT(m_pwm.num_rows>0 && m_pwm.num_cols>0)

	int32_t m_w_rows = CMath::pow(m_pwm.num_rows, m_pwm.num_cols);
	int32_t m_w_cols = num_pos;

	m_w = SGMatrix<float64_t>(m_w_cols,m_w_rows);

	uint8_t* window=SG_MALLOC(uint8_t, m_pwm.num_cols);
	SGVector<uint8_t>::fill_vector(window, m_pwm.num_cols, (uint8_t) 0);

	const int32_t last_idx=m_pwm.num_cols-1;
	for (int32_t i=0; i<m_w_rows; i++)
	{
		for (int32_t j=0; j<m_w_cols; j++)
			m_w[j*m_w_rows+i]=get_log_likelihood_window(window, m_pwm.num_cols, j);

		window[last_idx]++;
		int32_t window_ptr=last_idx;
		while (window[window_ptr]==m_pwm.num_rows && window_ptr>0)
		{
			window[window_ptr]=0;
			window_ptr--;
			window[window_ptr]++;
		}

	}
}

void CPositionalPWM::register_params()
{
	m_parameters->add(&m_poim, "poim", "POIM Scoring Matrix");
	m_parameters->add(&m_w, "w", "Scoring Matrix");
	m_parameters->add(&m_pwm, "pwm", "Positional Weight Matrix.");
	m_parameters->add(&m_sigma, "sigma", "Standard Deviation.");
	m_parameters->add(&m_mean, "mean", "Mean.");
}

void CPositionalPWM::compute_scoring(int32_t max_degree)
{
	int32_t num_feat=m_w.num_cols;
	int32_t num_sym=0;
	int32_t order=m_pwm.num_rows;
	int32_t num_words=m_pwm.num_cols;

	CAlphabet* alpha=new CAlphabet(DNA);
	CStringFeatures<uint16_t>* str= new CStringFeatures<uint16_t>(alpha);
	int32_t num_bits=alpha->get_num_bits();
	str->compute_symbol_mask_table(num_bits);

	for (int32_t i=0; i<order; i++)
		num_sym+=CMath::pow((int32_t) num_words,i+1);

	m_poim = SGVector<float64_t>(num_feat*num_sym);
	memset(m_poim.vector,0, size_t(num_feat)*size_t(num_sym));

	uint32_t kmer_mask=0;
	uint32_t words=CMath::pow((int32_t) num_words,(int32_t) order);
	int32_t offset=0;

	for (int32_t o=0; o<max_degree; o++)
	{
		float64_t* contrib=&m_poim[offset];
		offset+=CMath::pow((int32_t) num_words,(int32_t) o+1);

		kmer_mask=(kmer_mask<<(num_bits)) | str->get_masked_symbols(0xffff, 1);

		for (int32_t p=-o; p<order; p++)
		{
			int32_t o_sym=0, m_sym=0, il=0,ir=0, jl=0;
			uint32_t imer_mask=kmer_mask;
			uint32_t jmer_mask=kmer_mask;

			if (p<0)
			{
				il=-p;
				m_sym=order-o-p-1;
				o_sym=-p;
			}
			else if (p<order-o)
			{
				ir=p;
				m_sym=order-o-1;
			}
			else
			{
				ir=p;
				m_sym=p;
				o_sym=p-order+o+1;
				jl=order-ir;
				imer_mask=(kmer_mask>>(num_bits*o_sym));
				jmer_mask=(kmer_mask>>(num_bits*jl));
			}

			float64_t marginalizer=
				1.0/CMath::pow((int32_t) num_words,(int32_t) m_sym);

			for (uint32_t i=0; i<words; i++)
			{
				uint16_t x= ((i << (num_bits*il)) >> (num_bits*ir)) & imer_mask;

				if (p>=0 && p<order-o)
				{
					contrib[x]+=m_w[m_w.num_cols*ir+i]*marginalizer;
				}
				else
				{
					for (uint32_t j=0; j< (uint32_t) CMath::pow((int32_t) num_words, (int32_t) o_sym); j++)
					{
						uint32_t c=x | ((j & jmer_mask) << (num_bits*jl));
						contrib[c]+=m_w[m_w.num_cols*il+i]*marginalizer;
					}
				}
			}
		}
	}
}

SGMatrix<float64_t> CPositionalPWM::get_scoring(int32_t d)
{
	int32_t offs=0;
	for (int32_t i=0; i<d-1; i++)
		offs+=CMath::pow((int32_t) m_w.num_rows,i+1);
	int32_t rows=CMath::pow((int32_t) m_w.num_rows,d);
	int32_t cols=m_w.num_cols;
	float64_t* scoring_matrix = SG_MALLOC(float64_t, rows*cols);
	memcpy(scoring_matrix,m_poim.vector+offs,rows*cols*sizeof(float64_t));
	return SGMatrix<float64_t>(scoring_matrix,rows,cols);
}
