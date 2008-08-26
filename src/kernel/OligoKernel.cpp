#include "kernel/OligoKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/StringFeatures.h"

#include <map>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

COligoKernel::COligoKernel(INT cache_sz, INT kmer_len, DREAL w) : CStringKernel<CHAR>(cache_sz), k(kmer_len), width(w)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

COligoKernel::~COligoKernel()
{

}

bool COligoKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<CHAR>::init(l,r);
	return init_normalizer();
}

bool COligoKernel::cmpOligos_( pair<int, double> a, pair<int, double> b ) 
{
	return (a.second < b.second);
}

void COligoKernel::encodeOligo(const string& sequence,
		unsigned int k_mer_length,
		const string& allowed_characters,
		vector< pair<int, double> >& values)
{
	double oligo_value = 0.;
	double factor      = 1.;
	map<string::value_type, unsigned int> residue_values;
	unsigned int counter = 0;
	unsigned int number_of_residues = allowed_characters.size();
	unsigned int sequence_length = sequence.size();
	bool sequence_ok = true;

	// checking if sequence contains illegal characters
	for (unsigned int i = 0; i < sequence.size(); ++i)
	{
		if (allowed_characters.find(sequence.at(i)) == string::npos)
			sequence_ok = false;
	}

	if (sequence_ok && k_mer_length <= sequence_length)
	{	
		values.resize(sequence_length - k_mer_length + 1, pair<int, double>());
		for (unsigned int i = 0; i < number_of_residues; ++i)
		{	
			residue_values.insert(make_pair(allowed_characters[i], counter));
			++counter;
		}
		for (int k = k_mer_length - 1; k >= 0; k--)
		{
			oligo_value += factor * residue_values[sequence[k]];
			factor *= number_of_residues;
		}
		factor /= number_of_residues;	
		counter = 0;
		values[counter].first = 1;
		values[counter].second = oligo_value;
		++counter;

		for (unsigned int j = 1; j < sequence_length - k_mer_length + 1; j++)
		{
			oligo_value -= factor * residue_values[sequence[j - 1]];
			oligo_value = oligo_value * number_of_residues + residue_values[sequence[j + k_mer_length - 1]];

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

void COligoKernel::getSequences(const vector<string>& sequences, 
		unsigned int k_mer_length, 
		const string& allowed_characters, 
		vector< vector< pair<int, double> > >& encoded_sequences)
{
	vector< pair<int, double> > temp_vector;
	encoded_sequences.resize(sequences.size(), vector< pair<int, double> >());

	for (unsigned int i = 0; i < sequences.size(); ++i)
	{
		encodeOligo(sequences[i], k_mer_length, allowed_characters, temp_vector);
		encoded_sequences[i] = temp_vector;
	}
}

void COligoKernel::getExpFunctionCache(double sigma, unsigned int sequence_length, vector<double>& cache)
{
	cache.resize(sequence_length, 0.);
	cache[0] = 1;
	for (unsigned int i = 1; i < sequence_length - 1; i++)
	{
		cache[i] = exp((-1 / (4.0 * sigma * sigma)) * i * i);
	}
}

double COligoKernel::kernelOligoFast(const vector< pair<int, double> >& x, 
		const vector< pair<int, double> >& y,
		const vector<double>& gauss_table,
		int max_distance)
{
	double kernel = 0;
	int    i1     = 0;
	int    i2     = 0;
	int    c1     = 0;
	unsigned int x_size = x.size();
	unsigned int y_size = y.size();

	while ((unsigned int) i1 < x_size && (unsigned int) i2 < y_size)
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


double COligoKernel::kernelOligo(const vector< pair<int, double> >&    x, 
		const vector< pair<int, double> >&    y,
		double 			              sigma_square)
{
	double kernel = 0;
	int    i1     = 0;
	int    i2     = 0;
	int    c1     = 0;
	unsigned int x_size = x.size();
	unsigned int y_size = y.size();

	while ((unsigned int) i1 < x_size && (unsigned int) i2 < y_size)
	{
		if (x[i1].second == y[i2].second)
		{
			kernel += exp(-1 * (x[i1].first - y[i2].first) * (x[i1].first - y[i2].first) / (4 * sigma_square));

			if (((unsigned int) i1+1) < x_size && x[i1].second == x[i1 + 1].second)
			{
				i1++;
				c1++;
			}
			else if (((unsigned int) i2+1) <y_size && y[i2].second == y[i2 + 1].second)
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

DREAL COligoKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	CHAR* avec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);
	vector< pair<int,double> > aenc;
	vector< pair<int,double> > benc;
	encodeOligo(string(avec, alen), k, "ACGT", aenc);
	encodeOligo(string(bvec, alen), k, "ACGT", benc);
	DREAL result=kernelOligo(aenc, benc, width);
	return result;
}

