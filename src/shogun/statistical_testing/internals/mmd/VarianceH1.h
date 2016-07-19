/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef VARIANCE_H1__H_
#define VARIANCE_H1__H_

#include <vector>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

using std::vector;

namespace shogun
{

namespace internal
{

namespace mmd
{

struct VarianceH1
{
	VarianceH1() : m_should_free_terms(true)
	{
	}

	void init_terms()
	{
		m_sum_colwise_x.resize(m_n_x, 0);
		m_sum_colwise_y.resize(m_n_y, 0);
		m_sum_rowwise_xy.resize(m_n_y, 0);
		m_sum_colwise_xy.resize(m_n_x, 0);
		if (m_second_order_terms.rows()==m_n_x && m_second_order_terms.cols()==m_n_y)
			m_second_order_terms.setZero();
		else
			m_second_order_terms=Eigen::MatrixXd::Zero(m_n_x, m_n_x);

		m_sum_x=0;
		m_sum_y=0;
		m_sum_xy=0;
		m_sum_sq_x=0;
		m_sum_sq_y=0;
		m_sum_sq_xy=0;
	}

	void free_terms()
	{
		if (m_should_free_terms)
		{
			m_sum_colwise_x.resize(0);
			m_sum_colwise_y.resize(0);
			m_sum_rowwise_xy.resize(0);
			m_sum_colwise_xy.resize(0);
			m_second_order_terms=Eigen::MatrixXd::Zero(0, 0);
		}
	}

	template <typename T>
	void add_terms(T kernel_value, index_t i, index_t j)
	{
		if (i<m_n_x && j<m_n_x)
		{
			m_sum_x+=2*kernel_value;
			m_sum_sq_x+=2*kernel_value*kernel_value;
			m_sum_colwise_x[i]+=kernel_value;
			m_sum_colwise_x[j]+=kernel_value;
			m_second_order_terms(i, j)+=kernel_value;
			m_second_order_terms(j, i)+=kernel_value;
		}
		else if (i>=m_n_x && j>=m_n_x)
		{
			m_sum_y+=2*kernel_value;
			m_sum_sq_y+=2*kernel_value*kernel_value;
			m_sum_colwise_y[i-m_n_x]+=kernel_value;
			m_sum_colwise_y[j-m_n_x]+=kernel_value;
			m_second_order_terms(i-m_n_x, j-m_n_x)+=kernel_value;
			m_second_order_terms(j-m_n_x, i-m_n_x)+=kernel_value;
		}
		else if (i<m_n_x && j>=m_n_x)
		{
			m_sum_xy+=kernel_value;
			m_sum_sq_xy+=kernel_value*kernel_value;
			if (j-i!=m_n_x)
			{
				m_second_order_terms(i, j-m_n_x)-=kernel_value;
				m_second_order_terms(j-m_n_x, i)-=kernel_value;
			}
			m_sum_rowwise_xy[i]+=kernel_value;
			m_sum_colwise_xy[j-m_n_x]+=kernel_value;
		}
	}

	float64_t compute_variance_estimate()
	{
		Eigen::Map<Eigen::VectorXd> map_sum_colwise_x(m_sum_colwise_x.data(), m_sum_colwise_x.size());
		Eigen::Map<Eigen::VectorXd> map_sum_colwise_y(m_sum_colwise_y.data(), m_sum_colwise_y.size());
		Eigen::Map<Eigen::VectorXd> map_sum_rowwise_xy(m_sum_rowwise_xy.data(), m_sum_rowwise_xy.size());
		Eigen::Map<Eigen::VectorXd> map_sum_colwise_xy(m_sum_colwise_xy.data(), m_sum_colwise_xy.size());

		auto t_0=(map_sum_colwise_x.dot(map_sum_colwise_x)-m_sum_sq_x)/m_n_x/(m_n_x-1)/(m_n_x-2);
		auto t_1=CMath::sq(m_sum_x/m_n_x/(m_n_x-1));

		auto t_2=map_sum_colwise_x.dot(map_sum_rowwise_xy)*2/m_n_x/(m_n_x-1)/m_n_y;
		auto t_3=m_sum_x*m_sum_xy*2/m_n_x/m_n_x/(m_n_x-1)/m_n_y;

		auto t_4=(map_sum_colwise_y.dot(map_sum_colwise_y)-m_sum_sq_y)/m_n_y/(m_n_y-1)/(m_n_y-2);
		auto t_5=CMath::sq(m_sum_y/m_n_y/(m_n_y-1));

		auto t_6=map_sum_colwise_y.dot(map_sum_colwise_xy)*2/m_n_y/(m_n_y-1)/m_n_x;
		auto t_7=m_sum_y*m_sum_xy*2/m_n_y/m_n_y/(m_n_y-1)/m_n_x;

		auto t_8=(map_sum_rowwise_xy.dot(map_sum_rowwise_xy)-m_sum_sq_xy)/m_n_y/(m_n_y-1)/m_n_x;
		auto t_9=2*CMath::sq(m_sum_xy/m_n_x/m_n_y);
		auto t_10=(map_sum_colwise_xy.dot(map_sum_colwise_xy)-m_sum_sq_xy)/m_n_x/(m_n_x-1)/m_n_y;

		auto var_first=(t_0-t_1)-t_2+t_3+(t_4-t_5)-t_6+t_7+(t_8-t_9+t_10);
		var_first*=4.0*(m_n_x-2)/m_n_x/(m_n_x-1);

		auto var_second=2.0/m_n_x/m_n_y/(m_n_x-1)/(m_n_y-1)*m_second_order_terms.array().square().sum();

		auto variance_estimate=var_first+var_second;
		if (variance_estimate<0)
			variance_estimate=var_second;

		return variance_estimate;
	}

	template <class Kernel>
	float64_t operator()(const Kernel& kernel)
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		ASSERT(m_n_x==m_n_y);
		const index_t size=m_n_x+m_n_y;
		init_terms();
		for (auto j=0; j<size; ++j)
		{
			for (auto i=0; i<j; ++i)
				add_terms(kernel(i, j), i, j);
		}
		auto variance_estimate=compute_variance_estimate();
		free_terms();
		return variance_estimate;
	}

	index_t m_n_x;
	index_t m_n_y;

	float64_t m_sum_x;
	float64_t m_sum_y;
	float64_t m_sum_xy;
	float64_t m_sum_sq_x;
	float64_t m_sum_sq_y;
	float64_t m_sum_sq_xy;

	vector<float64_t> m_sum_colwise_x;
	vector<float64_t> m_sum_colwise_y;
	vector<float64_t> m_sum_rowwise_xy;
	vector<float64_t> m_sum_colwise_xy;
	Eigen::MatrixXd m_second_order_terms;

	bool m_should_free_terms;
};

}

}

}

#endif // VARIANCE_H1__H_
