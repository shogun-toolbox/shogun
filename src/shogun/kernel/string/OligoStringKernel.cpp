/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Bjoern Esser
 */
#include <shogun/kernel/string/OligoStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

#include <map>
#include <vector>
#include <algorithm>

using namespace shogun;

OligoStringKernel::OligoStringKernel()
  : StringKernel<char>()
{
	init();
}

OligoStringKernel::OligoStringKernel(int32_t cache_sz, int32_t kmer_len, float64_t w)
: StringKernel<char>(cache_sz)
{
	init();

	k=kmer_len;
	width=w;
}

OligoStringKernel::OligoStringKernel(
		const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
		int32_t kmer_len, float64_t w)
: StringKernel<char>()
{
	init();

	k=kmer_len;
	width=w;

	init(l, r);
}

OligoStringKernel::~OligoStringKernel()
{
	cleanup();
}

void OligoStringKernel::cleanup()
{
	gauss_table=SGVector<float64_t>();
	Kernel::cleanup();
}

bool OligoStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();

	StringKernel<char>::init(l,r);
	int32_t max_len=Math::max(
			std::static_pointer_cast<StringFeatures<char>>(l)->get_max_vector_length(),
			std::static_pointer_cast<StringFeatures<char>>(r)->get_max_vector_length()
			);

	require(k>0, "k must be >0");
	require(width>0, "width must be >0");

	getExpFunctionCache(max_len);
	return init_normalizer();
}

void OligoStringKernel::encodeOligo(
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

void OligoStringKernel::getSequences(
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

void OligoStringKernel::getExpFunctionCache(uint32_t sequence_length)
{
	gauss_table=SGVector<float64_t>(sequence_length);

	gauss_table[0] = 1;
	for (uint32_t i = 1; i < sequence_length; i++)
		gauss_table[i] = exp(-Math::sq((float64_t) i) / width);
}

float64_t OligoStringKernel::kernelOligoFast(
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

float64_t OligoStringKernel::kernelOligo(
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
			result += exp(-Math::sq(x[i1].first - y[i2].first) / width);

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

float64_t OligoStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_a, free_b;
	char* avec=std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_a);
	char* bvec=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_b);
	std::vector< std::pair<int32_t, float64_t> > aenc;
	std::vector< std::pair<int32_t, float64_t> > benc;
	encodeOligo(std::string(avec, alen), k, "ACGT", aenc);
	encodeOligo(std::string(bvec, alen), k, "ACGT", benc);
	//float64_t result=kernelOligo(aenc, benc);
	float64_t result=kernelOligoFast(aenc, benc);
	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_a);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_b);
	return result;
}

void OligoStringKernel::init()
{
	k=0;
	width=0.0;

	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());

	SG_ADD(&k, "k", "K-mer length.", ParameterProperties::HYPER);
	SG_ADD(&width, "width", "Width of Gaussian.", ParameterProperties::HYPER);
	SG_ADD(&gauss_table, "gauss_table", "Gauss Cache Table.");
}
