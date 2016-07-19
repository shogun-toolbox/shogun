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
	template <class Kernel>
	float64_t operator()(const Kernel& kernel)
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		ASSERT(m_n_x==m_n_y);
		const index_t size=m_n_x+m_n_y;

		sum_colwise_x.resize(m_n_x, 0);
		sum_colwise_y.resize(m_n_y, 0);
		sum_rowwise_xy.resize(m_n_y, 0);
		sum_colwise_xy.resize(m_n_x, 0);

		float64_t sum_x=0;
		float64_t sum_y=0;
		float64_t sum_xy=0;
		float64_t sum_sq_x=0;
		float64_t sum_sq_y=0;
		float64_t sum_sq_xy=0;

		Eigen::MatrixXd second_order_terms=Eigen::MatrixXd::Zero(m_n_x, m_n_x);

		for (auto j=0; j<size; ++j)
		{
			for (auto i=0; i<j; ++i)
			{
				auto kernel_value=kernel(i, j);
				if (i<m_n_x && j<m_n_x)
				{
					sum_x+=2*kernel_value;
					sum_sq_x+=2*kernel_value*kernel_value;
					sum_colwise_x[i]+=kernel_value;
					sum_colwise_x[j]+=kernel_value;
					second_order_terms(i, j)+=kernel_value;
					second_order_terms(j, i)+=kernel_value;
				}
				else if (i>=m_n_x && j>=m_n_x)
				{
					sum_y+=2*kernel_value;
					sum_sq_y+=2*kernel_value*kernel_value;
					sum_colwise_y[i-m_n_x]+=kernel_value;
					sum_colwise_y[j-m_n_x]+=kernel_value;
					second_order_terms(i-m_n_x, j-m_n_x)+=kernel_value;
					second_order_terms(j-m_n_x, i-m_n_x)+=kernel_value;
				}
				else if (i<m_n_x && j>=m_n_x)
				{
					sum_xy+=kernel_value;
					sum_sq_xy+=kernel_value*kernel_value;
					if (j-i!=m_n_x)
					{
						second_order_terms(i, j-m_n_x)-=kernel_value;
						second_order_terms(j-m_n_x, i)-=kernel_value;
					}
					sum_rowwise_xy[i]+=kernel_value;
					sum_colwise_xy[j-m_n_x]+=kernel_value;
				}
			}
		}

		typedef Eigen::Matrix<float64_t, Eigen::Dynamic, 1> VectorXt;
		Eigen::Map<const VectorXt> map_sum_colwise_x(sum_colwise_x.data(), sum_colwise_x.size());
		Eigen::Map<const VectorXt> map_sum_colwise_y(sum_colwise_y.data(), sum_colwise_y.size());
		Eigen::Map<const VectorXt> map_sum_rowwise_xy(sum_rowwise_xy.data(), sum_rowwise_xy.size());
		Eigen::Map<const VectorXt> map_sum_colwise_xy(sum_colwise_xy.data(), sum_colwise_xy.size());

		auto t_0=(map_sum_colwise_x.dot(map_sum_colwise_x)-sum_sq_x)/m_n_x/(m_n_x-1)/(m_n_x-2);
		auto t_1=CMath::sq(sum_x/m_n_x/(m_n_x-1));

		auto t_2=map_sum_colwise_x.dot(map_sum_rowwise_xy)*2/m_n_x/(m_n_x-1)/m_n_y;
		auto t_3=sum_x*sum_xy*2/m_n_x/m_n_x/(m_n_x-1)/m_n_y;

		auto t_4=(map_sum_colwise_y.dot(map_sum_colwise_y)-sum_sq_y)/m_n_y/(m_n_y-1)/(m_n_y-2);
		auto t_5=CMath::sq(sum_y/m_n_y/(m_n_y-1));

		auto t_6=map_sum_colwise_y.dot(map_sum_colwise_xy)*2/m_n_y/(m_n_y-1)/m_n_x;
		auto t_7=sum_y*sum_xy*2/m_n_y/m_n_y/(m_n_y-1)/m_n_x;

		auto t_8=(map_sum_rowwise_xy.dot(map_sum_rowwise_xy)-sum_sq_xy)/m_n_y/(m_n_y-1)/m_n_x;
		auto t_9=2*CMath::sq(sum_xy/m_n_x/m_n_y);
		auto t_10=(map_sum_colwise_xy.dot(map_sum_colwise_xy)-sum_sq_xy)/m_n_x/(m_n_x-1)/m_n_y;

		auto var_first=(t_0-t_1)-t_2+t_3+(t_4-t_5)-t_6+t_7+(t_8-t_9+t_10);
		var_first*=4.0*(m_n_x-2)/m_n_x/(m_n_x-1);

		auto var_second=2.0/m_n_x/m_n_y/(m_n_x-1)/(m_n_y-1)*second_order_terms.array().square().sum();

		auto variance_estimate=var_first+var_second;
		if (variance_estimate<0)
			variance_estimate=var_second;

		sum_colwise_x.resize(0);
		sum_colwise_y.resize(0);
		sum_rowwise_xy.resize(0);
		sum_colwise_xy.resize(0);

		return variance_estimate;
	}

	index_t m_n_x;
	index_t m_n_y;

	vector<float64_t> sum_colwise_x;
	vector<float64_t> sum_colwise_y;
	vector<float64_t> sum_rowwise_xy;
	vector<float64_t> sum_colwise_xy;
};

}

}

}

#endif // VARIANCE_H1__H_
