/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <vector>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Trie.h>
#include <shogun/base/Parallel.h>

#include <shogun/kernel/string/SpectrumMismatchRBFKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

#include <vector>
#include <string>

#include <assert.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

CSpectrumMismatchRBFKernel::CSpectrumMismatchRBFKernel() :
		CStringKernel<char>(0)
{
	init();
	register_params();
}

CSpectrumMismatchRBFKernel::CSpectrumMismatchRBFKernel(int32_t size,
		float64_t* AA_matrix_, int32_t nr, int32_t nc, int32_t degree_,
		int32_t max_mismatch_, float64_t width_) :
		CStringKernel<char>(size), alphabet(NULL), degree(degree_), max_mismatch(
				max_mismatch_), width(width_)
{
	init();
	target_letter_0=-1;
	set_AA_matrix(AA_matrix_, nr, nc);
	register_params();
}

CSpectrumMismatchRBFKernel::CSpectrumMismatchRBFKernel(CStringFeatures<char>* l,
		CStringFeatures<char>* r, int32_t size, float64_t* AA_matrix_,
		int32_t nr, int32_t nc, int32_t degree_, int32_t max_mismatch_,
		float64_t width_) :
		CStringKernel<char>(size), alphabet(NULL), degree(degree_), max_mismatch(
				max_mismatch_), width(width_)
{
	target_letter_0=-1;

	set_AA_matrix(AA_matrix_, nr, nc);
	init(l, r);
	register_params();
}

CSpectrumMismatchRBFKernel::~CSpectrumMismatchRBFKernel()
{
	cleanup();
	SG_UNREF(kernel_matrix);
}

bool CSpectrumMismatchRBFKernel::init(CFeatures* l, CFeatures* r)
{
	int32_t lhs_changed=(lhs!=l);
	int32_t rhs_changed=(rhs!=r);

	CStringKernel<char>::init(l, r);

	SG_DEBUG("lhs_changed: %i\n", lhs_changed)
	SG_DEBUG("rhs_changed: %i\n", rhs_changed)

	CStringFeatures<char>* sf_l=(CStringFeatures<char>*)l;
	CStringFeatures<char>* sf_r=(CStringFeatures<char>*)r;

	SG_UNREF(alphabet);
	alphabet=sf_l->get_alphabet();
	CAlphabet* ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties&=((uint64_t)(-1))^(KP_LINADD|KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())
	SG_UNREF(ralphabet);

	compute_all();

	return init_normalizer();
}

void CSpectrumMismatchRBFKernel::cleanup()
{

	SG_UNREF(alphabet);
	alphabet=NULL;

	CKernel::cleanup();
}

float64_t CSpectrumMismatchRBFKernel::AA_helper(std::string &path,
		const char* joint_seq, unsigned int index)
{
	float64_t diff=0.0;

	for (unsigned int i=0; i<path.size(); i++)
	{
		if (path[i]!=joint_seq[index+i])
		{
			diff+=AA_matrix.matrix[(path[i]-1)*128+path[i]-1];
			diff-=2*AA_matrix.matrix[(path[i]-1)*128+joint_seq[index+i]-1];
			diff+=AA_matrix.matrix[(joint_seq[index+i]-1)*128+joint_seq[index+i]
					-1];
		}
	}

	return exp(-diff/width);
}

void CSpectrumMismatchRBFKernel::compute_helper_all(const char *joint_seq,
		std::vector<struct joint_list_struct> &joint_list, std::string path,
		unsigned int d)
{
	const char* AA="ACDEFGHIKLMNPQRSTVWY";
	const unsigned int num_AA=strlen(AA);

	assert(path.size()==d);

	for (unsigned int i=0; i<num_AA; i++)
	{
		std::vector<struct joint_list_struct> joint_list_;

		if (d==0)
			SG_PRINT("i=%i: ", i);
		if (d==0&&target_letter_0!=-1&&(int)i!=target_letter_0)
			continue;

		if (d==1)
		{
			SG_PRINT("*");
		}
		if (d==2)
		{
			SG_PRINT("+");
		}

		for (unsigned int j=0; j<joint_list.size(); j++)
		{
			if (joint_seq[joint_list[j].index+d]!=AA[i])
			{
				if (joint_list[j].mismatch+1<=(unsigned int)max_mismatch)
				{
					struct joint_list_struct list_item;
					list_item=joint_list[j];
					list_item.mismatch=joint_list[j].mismatch+1;
					joint_list_.push_back(list_item);
				}
			}
			else
				joint_list_.push_back(joint_list[j]);
		}

		if (joint_list_.size()>0)
		{
			std::string path_=path+AA[i];

			if (d+1<(unsigned int)degree)
			{
				compute_helper_all(joint_seq, joint_list_, path_, d+1);
			}
			else
			{
				CDynamicArray<float64_t> feats;
				feats.resize_array(kernel_matrix->get_dim1());
				feats.set_const(0);

				for (unsigned int j=0; j<joint_list_.size(); j++)
				{
					if (width==0.0)
					{
						feats[joint_list_[j].ex_index]++;
						//if (joint_mismatch_[j]==0)
						//	feats[joint_ex_index_[j]]+=3 ;
					}
					else
					{
						if (joint_list_[j].mismatch!=0)
							feats[joint_list_[j].ex_index]+=AA_helper(path_,
									joint_seq, joint_list_[j].index);
						else
							feats[joint_list_[j].ex_index]++;
					}
				}

				std::vector<int> idx;
				for (int r=0; r<feats.get_array_size(); r++)
					if (feats[r]!=0.0)
						idx.push_back(r);

				for (unsigned int r=0; r<idx.size(); r++)
					for (unsigned int s=r; s<idx.size(); s++)
						if (s==r)
							kernel_matrix->set_element(
									feats[idx[r]]*feats[idx[s]]
											+kernel_matrix->get_element(idx[r],
													idx[s]), idx[r], idx[s]);
						else
						{
							kernel_matrix->set_element(
									feats[idx[r]]*feats[idx[s]]
											+kernel_matrix->get_element(idx[r],
													idx[s]), idx[r], idx[s]);
							kernel_matrix->set_element(
									feats[idx[r]]*feats[idx[s]]
											+kernel_matrix->get_element(idx[s],
													idx[r]), idx[s], idx[r]);
						}
			}
		}
		if (d==0)
			SG_PRINT("\n");
	}
}

void CSpectrumMismatchRBFKernel::compute_all()
{
	std::string joint_seq;
	std::vector<struct joint_list_struct> joint_list;

	assert(lhs->get_num_vectors()==rhs->get_num_vectors());
	kernel_matrix->resize_array(lhs->get_num_vectors(), lhs->get_num_vectors());
	kernel_matrix_length=lhs->get_num_vectors()*rhs->get_num_vectors();
	for (int i=0; i<lhs->get_num_vectors(); i++)
		for (int j=0; j<lhs->get_num_vectors(); j++)
			kernel_matrix->set_element(0, i, j);

	for (int i=0; i<lhs->get_num_vectors(); i++)
	{
		int32_t alen;
		bool free_avec;
		char* avec=((CStringFeatures<char>*)lhs)->get_feature_vector(i, alen,
				free_avec);

		for (int apos=0; apos+degree-1<alen; apos++)
		{
			struct joint_list_struct list_item;
			list_item.ex_index=i;
			list_item.index=apos+joint_seq.size();
			list_item.mismatch=0;

			joint_list.push_back(list_item);
		}
		joint_seq+=std::string(avec, alen);

		((CStringFeatures<char>*)lhs)->free_feature_vector(avec, i, free_avec);
	}

	compute_helper_all(joint_seq.c_str(), joint_list, "", 0);
}

float64_t CSpectrumMismatchRBFKernel::compute(int32_t idx_a, int32_t idx_b)
{
	return kernel_matrix->element(idx_a, idx_b);
}

bool CSpectrumMismatchRBFKernel::set_AA_matrix(float64_t* AA_matrix_,
		int32_t nr, int32_t nc)
{
	if (AA_matrix_)
	{
		if (nr!=128 || nc!=128)
			SG_ERROR("AA_matrix should be of shape 128x128\n")

		AA_matrix=SGMatrix<float64_t>(nc, nr);
		SG_DEBUG("Setting AA_matrix\n")
		memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t));
		return true;
	}

	return false;
}

bool CSpectrumMismatchRBFKernel::set_max_mismatch(int32_t max)
{
	max_mismatch=max;

	if (lhs!=NULL&&rhs!=NULL)
		return init(lhs, rhs);
	else
		return true;
}

void CSpectrumMismatchRBFKernel::register_params()
{
	SG_ADD(&degree, "degree", "degree of the kernel", MS_AVAILABLE);
	SG_ADD(&AA_matrix, "AA_matrix", "128*128 scalar product matrix",
			MS_NOT_AVAILABLE);
	SG_ADD(&width, "width", "width of Gaussian", MS_AVAILABLE);
	SG_ADD(&target_letter_0, "target_letter_0", "target letter 0",
			MS_NOT_AVAILABLE);
	SG_ADD(&initialized, "initialized", "the mark of initialization status",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** )&kernel_matrix, "kernel_matrix",
			"the kernel matrix with its length "
					"defined by the number of vectors of the string features",
			MS_NOT_AVAILABLE);
}

void CSpectrumMismatchRBFKernel::register_alphabet()
{
	SG_ADD((CSGObject** )&alphabet, "alphabet", "the alphabet used by kernel",
			MS_NOT_AVAILABLE);
}

void CSpectrumMismatchRBFKernel::init()
{
	alphabet=NULL;
	degree=0;
	max_mismatch=0;
	width=0.0;
	kernel_matrix=new CDynamicArray<float64_t>();
	initialized=false;
	target_letter_0=0;
}

