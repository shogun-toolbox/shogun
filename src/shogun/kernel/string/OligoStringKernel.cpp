/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Christian Igel, Tobias Glasmachers
 * Copyright (C) 2008 Christian Igel, Tobias Glasmachers
 *
 * Shogun adjustments (W) 2008-2009,2013 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2013 Soeren Sonnenburg
 *
 */
#include <kernel/string/OligoStringKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <features/StringFeatures.h>

#include <map>
#include <vector>
#include <algorithm>

using namespace shogun;

COligoStringKernel::COligoStringKernel()
  : CStringKernel<char>()
{
	init();
}

COligoStringKernel::COligoStringKernel(int32_t cache_sz, int32_t kmer_len, float64_t w)
: CStringKernel<char>(cache_sz)
{
	init();

	k=kmer_len;
	width=w;
}

COligoStringKernel::COligoStringKernel(
		CStringFeatures<char>* l, CStringFeatures<char>* r,
		int32_t kmer_len, float64_t w)
: CStringKernel<char>()
{
	init();

	k=kmer_len;
	width=w;

	init(l, r);
}

COligoStringKernel::~COligoStringKernel()
{
	cleanup();
}

void COligoStringKernel::cleanup()
{
	gauss_table=SGVector<float64_t>();
	CKernel::cleanup();
}

bool COligoStringKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();

	CStringKernel<char>::init(l,r);
	int32_t max_len=CMath::max(
			((CStringFeatures<char>*) l)->get_max_vector_length(),
			((CStringFeatures<char>*) r)->get_max_vector_length()
			);

	REQUIRE(k>0, "k must be >0\n")
	REQUIRE(width>0, "width must be >0\n")

	getExpFunctionCache(max_len);
	return init_normalizer();
}

void COligoStringKernel::encodeOligo(
	const std::string& sequence, uint32_t k_mer_length,
	const std::string& allowed_characters,
	std::vector< std::pair<int32_t, float64_t> >& values)
{
	float64_t oligo_value = 0.;
	float64_t factor      = 1.;
	std::map<std::string::value_type, uint32_t> residue_values;
	uint32_t counter = 0;
	uint32_t number_of_residues = allowed_characters.size();
	uint32_t sequence_length = sequence.size();
	bool sequence_ok = true;

	// checking if sequence contains illegal characters
	for (uint32_t i = 0; i < sequence.size(); ++i)
	{
		if (allowed_characters.find(sequence.at(i)) == std::string::npos)
			sequence_ok = false;
	}

	if (sequence_ok && k_mer_length <= sequence_length)
	{
		values.resize(sequence_length - k_mer_length + 1,
			std::pair<int32_t, float64_t>());
		for (uint32_t i = 0; i < number_of_residues; ++i)
		{
			residue_values.insert(std::make_pair(allowed_characters[i], counter));
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

void COligoStringKernel::getSequences(
	const std::vector<std::string>& sequences, uint32_t k_mer_length,
	const std::string& allowed_characters,
	std::vector< std::vector< std::pair<int32_t, float64_t> > >& encoded_sequences)
{
	std::vector< std::pair<int32_t, float64_t> > temp_vector;
	encoded_sequences.resize(sequences.size(),
		std::vector< std::pair<int32_t, float64_t> >());

	for (uint32_t i = 0; i < sequences.size(); ++i)
	{
		encodeOligo(sequences[i], k_mer_length, allowed_characters, temp_vector);
		encoded_sequences[i] = temp_vector;
	}
}

void COligoStringKernel::getExpFunctionCache(uint32_t sequence_length)
{
	gauss_table=SGVector<float64_t>(sequence_length);

	gauss_table[0] = 1;
	for (uint32_t i = 1; i < sequence_length; i++)
		gauss_table[i] = exp(-CMath::sq((float64_t) i) / width);
}

float64_t COligoStringKernel::kernelOligoFast(
	const std::vector< std::pair<int32_t, float64_t> >& x,
	const std::vector< std::pair<int32_t, float64_t> >& y,
	int32_t max_distance)
{
	float64_t result = 0;
	int32_t  i1     = 0;
	int32_t  i2     = 0;
	int32_t  c1     = 0;
	uint32_t x_size = x.size();
	uint32_t y_size = y.size();

	while ((uint32_t) i1 + 1 < x_size && (uint32_t) i2 + 1 < y_size)
	{
		if (x[i1].second == y[i2].second)
		{
			if (max_distance < 0
					|| (abs(x[i1].first - y[i2].first)) <= max_distance)
			{
				result += gauss_table[abs((x[i1].first - y[i2].first))];
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
						while (y[i2].second == y[i2].second)
							i2++;
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
	return result;
}

float64_t COligoStringKernel::kernelOligo(
		const std::vector< std::pair<int32_t, float64_t> >&    x,
		const std::vector< std::pair<int32_t, float64_t> >&    y)
{
	float64_t result = 0;
	int32_t    i1     = 0;
	int32_t    i2     = 0;
	int32_t    c1     = 0;
	uint32_t x_size = x.size();
	uint32_t y_size = y.size();

	while ((uint32_t) i1 < x_size && (uint32_t) i2 < y_size)
	{
		if (x[i1].second == y[i2].second)
		{
			result += exp(-CMath::sq(x[i1].first - y[i2].first) / width);

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
	return result;
}

float64_t COligoStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_a, free_b;
	char* avec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_a);
	char* bvec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_b);
	std::vector< std::pair<int32_t, float64_t> > aenc;
	std::vector< std::pair<int32_t, float64_t> > benc;
	encodeOligo(std::string(avec, alen), k, "ACGT", aenc);
	encodeOligo(std::string(bvec, alen), k, "ACGT", benc);
	//float64_t result=kernelOligo(aenc, benc);
	float64_t result=kernelOligoFast(aenc, benc);
	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_a);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_b);
	return result;
}

void COligoStringKernel::init()
{
	k=0;
	width=0.0;

	set_normalizer(new CSqrtDiagKernelNormalizer());

	SG_ADD(&k, "k", "K-mer length.", MS_AVAILABLE);
	SG_ADD(&width, "width", "Width of Gaussian.", MS_AVAILABLE);
	SG_ADD(&gauss_table, "gauss_table", "Gauss Cache Table.", MS_NOT_AVAILABLE);
}
