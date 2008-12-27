/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Christian Igel, Tobias Glasmachers
 * Copyright (C) 2008 Christian Igel, Tobias Glasmachers
 *
 * Shogun adjustments (w) 2008 Soeren Sonnenburg
 */
#include "kernel/OligoKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/StringFeatures.h"

#include <map>
#include <vector>
#include <algorithm>

using namespace std;

COligoKernel::COligoKernel(int32_t cache_sz, int32_t kmer_len, float64_t w)
: CStringKernel<char>(cache_sz), k(kmer_len), width(w)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

COligoKernel::~COligoKernel()
{

}

bool COligoKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l,r);
	return init_normalizer();
}

bool COligoKernel::cmpOligos_(
	pair<int32_t, float64_t> a, pair<int32_t, float64_t> b)
{
	return (a.second < b.second);
}

void COligoKernel::encodeOligo(
	const string& sequence, uint32_t k_mer_length,
	const string& allowed_characters,
	vector< pair<int32_t, float64_t> >& values)
{
	float64_t oligo_value = 0.;
	float64_t factor      = 1.;
	map<string::value_type, uint32_t> residue_values;
	uint32_t counter = 0;
	uint32_t number_of_residues = allowed_characters.size();
	uint32_t sequence_length = sequence.size();
	bool sequence_ok = true;

	// checking if sequence contains illegal characters
	for (uint32_t i = 0; i < sequence.size(); ++i)
	{
		if (allowed_characters.find(sequence.at(i)) == string::npos)
			sequence_ok = false;
	}

	if (sequence_ok && k_mer_length <= sequence_length)
	{
		values.resize(sequence_length - k_mer_length + 1,
			pair<int32_t, float64_t>());
		for (uint32_t i = 0; i < number_of_residues; ++i)
		{	
			residue_values.insert(make_pair(allowed_characters[i], counter));
			++counter;
		}
		for (int32_t k = k_mer_length - 1; k >= 0; k--)
		{
			oligo_value += factor * residue_values[sequence[k]];
			factor *= number_of_residues;
		}
		factor /= number_of_residues;
		counter = 0;
		values[counter].first = 1;
		values[counter].second = oligo_value;
		++counter;

		for (uint32_t j = 1; j < sequence_length - k_mer_length + 1; j++)
		{
			oligo_value -= factor * residue_values[sequence[j - 1]];
			oligo_value = oligo_value * number_of_residues +
				residue_values[sequence[j + k_mer_length - 1]];

			values[counter].first = j + 1;
			values[counter].second = oligo_value ;
			++counter;
		}
		stable_sort(values.begin(), values.end(), cmpOligos_);
	}
	else
	{
		values.clear();
	}	
}

void COligoKernel::getSequences(
	const vector<string>& sequences, uint32_t k_mer_length,
	const string& allowed_characters,
	vector< vector< pair<int32_t, float64_t> > >& encoded_sequences)
{
	vector< pair<int32_t, float64_t> > temp_vector;
	encoded_sequences.resize(sequences.size(),
		vector< pair<int32_t, float64_t> >());

	for (uint32_t i = 0; i < sequences.size(); ++i)
	{
		encodeOligo(sequences[i], k_mer_length, allowed_characters, temp_vector);
		encoded_sequences[i] = temp_vector;
	}
}

void COligoKernel::getExpFunctionCache(
	float64_t sigma, uint32_t sequence_length, vector<float64_t>& cache)
{
	cache.resize(sequence_length, 0.);
	cache[0] = 1;
	for (uint32_t i = 1; i < sequence_length - 1; i++)
	{
		cache[i] = exp((-1 / (4.0 * sigma * sigma)) * i * i);
	}
}

float64_t COligoKernel::kernelOligoFast(
	const vector< pair<int32_t, float64_t> >& x,
	const vector< pair<int32_t, float64_t> >& y,
	const vector<float64_t>& gauss_table, int32_t max_distance)
{
	float64_t kernel = 0;
	int32_t  i1     = 0;
	int32_t  i2     = 0;
	int32_t  c1     = 0;
	uint32_t x_size = x.size();
	uint32_t y_size = y.size();

	while ((uint32_t) i1 < x_size && (uint32_t) i2 < y_size)
	{
		if (x[i1].second == y[i2].second)
		{
			if (max_distance < 0
					|| (abs(x[i1].first - y[i2].first)) <= max_distance)
			{
				kernel += gauss_table.at(abs((x[i1].first - y[i2].first)));
				if (x[i1].second == x[i1 + 1].second)
				{
					i1++;
					c1++;
				}
				else if (y[i2].second == y[i2 + 1].second)
				{
					i2++;
					i1 -= c1;
					c1 = 0;
				}
				else
				{
					i1++;
					i2++;
				}
			}
			else
			{
				if (x[i1].first < y[i2].first)
				{
					if (x[i1].second == x[i1 + 1].second)
					{
						i1++;
					}
					else if (y[i2].second == y[i2 + 1].second)
					{
						while(y[i2++].second == y[i2].second)
						{
							;
						}
						++i1;
						c1 = 0;
					}
					else
					{
						i1++;
						i2++;
						c1 = 0;
					}
				}
				else
				{
					i2++;
					i1 -= c1;
					c1 = 0;
				}
			}
		}
		else
		{
			if (x[i1].second < y[i2].second)
				i1++;
			else
				i2++;
			c1 = 0;
		}
	}
	return kernel;
}		


float64_t COligoKernel::kernelOligo(
		const vector< pair<int32_t, float64_t> >& x,
		const vector< pair<int32_t, float64_t> >& y,
		float64_t sigma_square)
{
	float64_t   kernel = 0;
	int32_t  i1     = 0;
	int32_t  i2     = 0;
	int32_t  c1     = 0;
	uint32_t x_size = x.size();
	uint32_t y_size = y.size();

	while ((uint32_t) i1 < x_size && (uint32_t) i2 < y_size)
	{
		if (x[i1].second == y[i2].second)
		{
			kernel += exp(-1 * (x[i1].first - y[i2].first) * (x[i1].first - y[i2].first) / (4 * sigma_square));

			if (((uint32_t) i1+1) < x_size && x[i1].second == x[i1 + 1].second)
			{
				i1++;
				c1++;
			}
			else if (((uint32_t) i2+1) <y_size && y[i2].second == y[i2 + 1].second)
			{
				i2++;
				i1 -= c1;
				c1 = 0;
			}
			else
			{
				i1++;
				i2++;
			}
		}
		else
		{
			if (x[i1].second < y[i2].second)
				i1++;
			else
				i2++;

			c1 = 0;
		}
	}
	return kernel;
}		

float64_t COligoKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	char* avec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen);
	char* bvec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen);
	vector< pair<int32_t, float64_t> > aenc;
	vector< pair<int32_t, float64_t> > benc;
	encodeOligo(string(avec, alen), k, "ACGT", aenc);
	encodeOligo(string(bvec, alen), k, "ACGT", benc);
	float64_t result=kernelOligo(aenc, benc, width);
	return result;
}

