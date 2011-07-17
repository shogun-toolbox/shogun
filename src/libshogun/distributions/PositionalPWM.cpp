/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include "distributions/PositionalPWM.h"
#include "lib/Mathematics.h"
#include "base/Parameter.h"
#include "features/Alphabet.h"
#include "features/StringFeatures.h"

using namespace shogun;

CPositionalPWM::CPositionalPWM() : CDistribution(),
	m_pwm_rows(0), m_pwm_cols(0), m_pwm(NULL),
	m_sigma(0), m_mean(0),
	m_w_rows(0), m_w_cols(0), m_w(NULL)

{
	register_params();
}

CPositionalPWM::~CPositionalPWM()
{
	delete[] m_pwm;
	delete[] m_w;
}

bool CPositionalPWM::train(CFeatures* data)
{
	SG_NOTIMPLEMENTED;
	return true;
}

int32_t CPositionalPWM::get_num_model_parameters()
{
	return m_pwm_rows*m_pwm_cols+2;
}

float64_t CPositionalPWM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param>0 && num_param<=m_pwm_rows*m_pwm_cols+2);

	if (num_param<m_pwm_rows*m_pwm_cols)
	{
		ASSERT(m_pwm);
		return m_pwm[num_param];
	}
	else if (num_param<m_pwm_rows*m_pwm_cols+1)
		return CMath::log(m_sigma);
	else
		return CMath::log(m_mean);
}

float64_t CPositionalPWM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CPositionalPWM::get_log_likelihood_example(int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CPositionalPWM::get_log_likelihood_window(uint8_t* window, int32_t len, float64_t pos)
{
	ASSERT(m_pwm_cols == len);
	float64_t score = CMath::log(1/(m_sigma*CMath::sqrt(2*M_PI))) -
			CMath::sq(pos-m_mean)/(2*CMath::sq(m_sigma));

	for (int32_t i=0; i<m_pwm_cols; i++)
		score+=m_pwm[m_pwm_rows*i+window[i]];

	return score;
}

void CPositionalPWM::compute_w(int32_t num_pos)
{
	m_w_rows=CMath::pow(m_pwm_rows, m_pwm_cols);
	m_w_cols=num_pos;

	delete[] m_w;
	m_w=new float64_t[m_w_cols*m_w_rows];

	uint8_t* window=new uint8_t[m_pwm_cols];
	CMath::fill_vector(window, m_pwm_cols, (uint8_t) 0);

	const int32_t last_idx=m_pwm_cols-1;
	for (int32_t i=0; i<m_w_rows; i++)
	{
		for (int32_t j=0; j<m_w_cols; j++)
			m_w[j*m_w_rows+i]=get_log_likelihood_window(window, m_pwm_cols, j);

		window[last_idx]++;
		int32_t window_ptr=last_idx;
		while (window[window_ptr]==m_pwm_rows && window_ptr>0)
		{
			window[window_ptr]=0;
			window_ptr--;
			window[window_ptr]++;
		}

	}

/*	CMath::fill_vector(m_w, m_w_rows*m_w_cols, 0.0);
	for (int32_t j=0; j<4; j++)
	{
		for (int32_t i=10; i<15; i++)
		{
			m_w[i*m_w_rows+j]=10;
		}
	}
*/
}

void CPositionalPWM::register_params()
{
	m_parameters->add_matrix(&m_w, &m_w_rows, &m_w_cols, "w", "Scoring Matrix");
	m_parameters->add_matrix(&m_pwm, &m_pwm_rows, &m_pwm_cols, "pwm", "Positional Weight Matrix.");
	m_parameters->add(&m_sigma, "sigma", "Standard Deviation.");
	m_parameters->add(&m_mean, "mean", "Mean.");
}

void CPositionalPWM::compute_scoring(int32_t max_degree)
{
	ASSERT(m_w);

	int32_t num_feat=m_w_cols;
	int32_t num_sym=0;
	int32_t order=m_pwm_cols;
	int32_t num_words=m_pwm_cols;
	float64_t* target=new float64_t[num_feat*num_sym];
	CAlphabet* alpha=new CAlphabet(DNA);
	CStringFeatures<uint16_t>* str= new CStringFeatures<uint16_t>();
	int32_t num_bits=alpha->get_num_bits();
	
	for (int32_t i=0; i<order; i++)
		num_sym+=CMath::pow((int32_t) num_words,i+1);

	uint32_t kmer_mask=0;
	uint32_t words=CMath::pow((int32_t) num_words,(int32_t) order);
	int32_t offset=0;

	for (int32_t o=0; o<max_degree; o++)
	{
		float64_t* contrib=&target[offset];
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
					contrib[x]+=m_w[i]*marginalizer;
				}
				else
				{
					for (uint32_t j=0; j< (uint32_t) CMath::pow((int32_t) num_words, (int32_t) o_sym); j++)
					{
						uint32_t c=x | ((j & jmer_mask) << (num_bits*jl));
						contrib[c]+=m_w[i]*marginalizer;
					}
				}
			}
		}
	}
}
