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

#include <lib/common.h>
#include <io/SGIO.h>
#include <lib/Signal.h>
#include <lib/Trie.h>
#include <base/Parallel.h>

#include <kernel/string/SpectrumRBFKernel.h>
#include <features/Features.h>
#include <features/StringFeatures.h>
#include <lib/SGStringList.h>
#include <math.h>

#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include <assert.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif


using namespace shogun;

CSpectrumRBFKernel::CSpectrumRBFKernel()
  : CStringKernel<char>(0)
{
    init();
	register_param();
}

CSpectrumRBFKernel::CSpectrumRBFKernel (int32_t size, float64_t *AA_matrix_, int32_t degree_, float64_t width_)
  : CStringKernel<char>(size), alphabet(NULL), degree(degree_), width(width_), sequences(NULL), string_features(NULL), nof_sequences(0), max_sequence_length(0)
{
	init();
	register_param();

	target_letter_0=-1 ;

	AA_matrix=SGMatrix<float64_t>(128,128);

	memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;

	read_profiles_and_sequences();
	SGStringList<char> string_list;
	string_list.strings = sequences;
	string_list.num_strings = nof_sequences;
	string_list.max_string_length = max_sequence_length;

	//string_features = new CStringFeatures<char>(sequences, nof_sequences, max_sequence_length, PROTEIN);
	string_features = new CStringFeatures<char>(string_list, IUPAC_AMINO_ACID);
	SG_REF(string_features)
	init(string_features, string_features);
}

CSpectrumRBFKernel::CSpectrumRBFKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t size, float64_t* AA_matrix_, int32_t degree_, float64_t width_)
: CStringKernel<char>(size), alphabet(NULL), degree(degree_), width(width_), sequences(NULL), string_features(NULL), nof_sequences(0), max_sequence_length(0)
{
	target_letter_0=-1 ;

	AA_matrix=SGMatrix<float64_t>(128,128);
	memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;

	init(l, r);
	register_param();
}

CSpectrumRBFKernel::~CSpectrumRBFKernel()
{
	cleanup();
	SG_UNREF(string_features);
	SG_FREE(sequences);
}

void CSpectrumRBFKernel::read_profiles_and_sequences()
{

		int32_t aa_to_index[128];//profile
		aa_to_index[(uint8_t) 'A'] = 0;
		aa_to_index[(uint8_t) 'R'] = 1;
		aa_to_index[(uint8_t) 'N'] = 2;
		aa_to_index[(uint8_t) 'D'] = 3;
		aa_to_index[(uint8_t) 'C'] = 4;
		aa_to_index[(uint8_t) 'Q'] = 5;
		aa_to_index[(uint8_t) 'E'] = 6;
		aa_to_index[(uint8_t) 'G'] = 7;
		aa_to_index[(uint8_t) 'H'] = 8;
		aa_to_index[(uint8_t) 'I'] = 9;
		aa_to_index[(uint8_t) 'L'] = 10;
		aa_to_index[(uint8_t) 'K'] = 11;
		aa_to_index[(uint8_t) 'M'] = 12;
		aa_to_index[(uint8_t) 'F'] = 13;
		aa_to_index[(uint8_t) 'P'] = 14;
		aa_to_index[(uint8_t) 'S'] = 15;
		aa_to_index[(uint8_t) 'T'] = 16;
		aa_to_index[(uint8_t) 'W'] = 17;
		aa_to_index[(uint8_t) 'Y'] = 18;
		aa_to_index[(uint8_t) 'V'] = 19;
	SG_DEBUG("initializing background\n")
	double background[20]; // profile
	background[0]=0.0799912015849807; //A
	background[1]=0.0484482507611578;//R
	background[2]=0.044293531582512;//N
	background[3]=0.0578891399707563;//D
	background[4]=0.0171846021407367;//C
	background[5]=0.0380578923048682;//Q
	background[6]=0.0638169929675978;//E
	background[7]=0.0760659374742852;//G
	background[8]=0.0223465499452473;//H
	background[9]=0.0550905793661343;//I
	background[10]=0.0866897071203864;//L
	background[11]=0.060458245507428;//K
	background[12]=0.0215379186368154;//M
	background[13]=0.0396348024787477;//F
	background[14]=0.0465746314476874;//P
	background[15]=0.0630028230885602;//S
	background[16]=0.0580394726014824;//T
	background[17]=0.0144991866213453;//W
	background[18]=0.03635438623143;//Y
	background[19]=0.0700241481678408;//V


	std::vector<std::string> seqs;
	//int32_t nof_sequences = 7329;

	double C = 0.8;
	const char *filename="/fml/ag-raetsch/home/toussaint/scp/aawd_compbio_workshop/code_nora/data/profile/profiles";
	std::ifstream fin(filename);

	SG_DEBUG("Reading profiles from %s\n", filename)
	std::string line;
	while (!fin.eof())
	{
		std::getline(fin, line);

		if (line[0] == '>') // new sequence
		{
			int idx = line.find_first_of(' ');
			sequence_labels.push_back(line.substr(1,idx-1));
			std::getline(fin, line);
			std::string orig_sequence = line;
			std::string sequence="";

			int len_line = line.length();

			// skip 3 lines

			std::getline(fin, line);
			std::getline(fin, line);
			std::getline(fin, line);

			profiles.push_back(std::vector<double>());

			std::vector<double>& curr_profile = profiles.back();
			for (int i=0; i < len_line; ++i)
			{
					std::getline(fin, line);
					int a = line.find_first_not_of(' '); // index position
					int b = line.find_first_of(' ', a); // index position
					a = line.find_first_not_of(' ', b); // aa position
					b = line.find_first_of(' ', a); // aa position
					std::string aa=line.substr(a,b-a);
					if (0) //(aa =="B" || aa == "X" || aa == "Z")
					{
						int pos = seqs.size()+1;
						SG_DEBUG("Skipping aa in sequence %d\n", pos)
				    continue;
	        }
					else
					{
						sequence += aa;

						a = line.find_first_not_of(' ', b); // beginning of block to ignore
						b = line.find_first_of(' ', a); // aa position

						for (int j=0; j < 19; ++j)
						{
							a = line.find_first_not_of(' ', b);
							b = line.find_first_of(' ', a);
						}

						int all_zeros = 1;
						// interesting block
						for (int j=0; j < 20; ++j)
						{
							a = line.find_first_not_of(' ', b);
							b = line.find_first_of(' ', a);
							double p = atof(line.substr(a, b-a).c_str());
							if (p > 0)
							{
								all_zeros = 0;
							}
							double value = -1* std::log(C*(p/100)+(1-C)*background[j]); // taken from Leslie's example, C actually corresponds to 1/(1+C)
							curr_profile.push_back(value);
							//SG_DEBUG("seq %d aa %d value %f p %f  bg %f\n", i, j, value,p, background[j])
						}

						if (all_zeros)
						{
							SG_DEBUG(">>>>>>>>>>>>>>> all zeros")
							if (aa !="B" && aa != "X" && aa != "Z")
							{
								//profile[i][temp_profile_index]=-log(C+(1-C)*background[re_candidate[temp_profile_index]]);
								int32_t aa_index = aa_to_index[(int)aa.c_str()[0]];
								double value = -1* std::log(C+(1-C)*background[aa_index]); // taken from Leslie's example, C actually corresponds to 1/(1+C)
								SG_DEBUG("before %f\n", profiles.back()[(i-1) * 20 + aa_index])
								curr_profile[(i*20) + aa_index] = value;
								SG_DEBUG(">>> aa %c \t %d \t %f\n", aa.c_str()[0], aa_index, value)

								/*
								for (int z=0; z <20; ++z)
								{
									SG_DEBUG(" %d \t %f\t", z, curr_profile[z])
								}
								SG_DEBUG("\n")
								*/
							}
						}
					}
			}

			if (curr_profile.size() != 20 * sequence.length())
	    {
				SG_ERROR("Something's wrong with the profile.\n")
				break;
			}

			seqs.push_back(sequence);


			/*
			// 6 irrelevant lines
			for (int i=0; i < 6; ++i)
			{
				std::getline(fin, line);
			}
			//
			*/
		}
	}

	fin.close();

	nof_sequences = seqs.size();
	sequences = SG_MALLOC(SGString<char>, nof_sequences);

	int max_len = 0;
	for (int i=0; i < nof_sequences; ++i)
	{
		int len = seqs[i].length();
		sequences[i].string = SG_MALLOC(char, len+1);
		sequences[i].slen = len;
		strcpy(sequences[i].string, seqs[i].c_str());

		if (len > max_len) max_len = len;
	}

	max_sequence_length = max_len;
	//string_features = new CStringFeatures<char>(sequences, nof_sequences, max_sequence_length, PROTEIN);

}

bool CSpectrumRBFKernel::init(CFeatures* l, CFeatures* r)
{
	// >> profile
/*
	read_profiles_and_sequences();
	l = string_features;
	r = string_features;
	*/
	// << profile

	int32_t lhs_changed=(lhs!=l);
	int32_t rhs_changed=(rhs!=r);

	CStringKernel<char>::init(l,r);

	SG_DEBUG("lhs_changed: %i\n", lhs_changed)
	SG_DEBUG("rhs_changed: %i\n", rhs_changed)

	CStringFeatures<char>* sf_l=(CStringFeatures<char>*) l;
	CStringFeatures<char>* sf_r=(CStringFeatures<char>*) r;

	SG_UNREF(alphabet);
	alphabet=sf_l->get_alphabet();
	CAlphabet* ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties &= ((uint64_t) (-1)) ^ (KP_LINADD | KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())
	SG_UNREF(ralphabet);


	return init_normalizer();
}

void CSpectrumRBFKernel::cleanup()
{

	SG_UNREF(alphabet);
	alphabet=NULL;

	CKernel::cleanup();
}

inline bool isaa(char c)
{
  if (c<65 || c>89 || c=='B' || c=='J' || c=='O' || c=='U' || c=='X' || c=='Z')
    return false ;
  return true ;
}

float64_t CSpectrumRBFKernel::AA_helper(const char* path, const int seq_degree, const char* joint_seq, unsigned int index)
{
	//const char* AA = "ARNDCQEGHILKMFPSTWYV";
  float64_t diff=0.0 ;

  for (int i=0; i<seq_degree; i++)
    {
      if (!isaa(path[i])||!isaa(joint_seq[index+i]))
	diff+=1.4 ;
      else
	{
	  diff += AA_matrix.matrix[ (path[i]-1)*128 + path[i] - 1] ;
	  diff -= 2*AA_matrix.matrix[ (path[i]-1)*128 + joint_seq[index+i] - 1] ;
	  diff += AA_matrix.matrix[ (joint_seq[index+i]-1)*128 + joint_seq[index+i] - 1] ;
	  if (CMath::is_nan(diff))
	    fprintf(stderr, "nan occurred: '%c' '%c'\n", path[i], joint_seq[index+i]) ;
	}
    }

  return exp( - diff/width) ;
}

float64_t CSpectrumRBFKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, afree);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
	  {
	    for (int32_t j=0; j<blen; j++)
	      {
		if ((i+degree<=alen) && (j+degree<=blen))
		  result += AA_helper(&(avec[i]), degree, bvec, j) ;
	      }
	  }

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	return result;
}

bool CSpectrumRBFKernel::set_AA_matrix(
	float64_t* AA_matrix_)
{

	if (AA_matrix_)
	{
		SG_DEBUG("Setting AA_matrix\n")
		memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;
		return true ;
	}

	return false;
}

void CSpectrumRBFKernel::register_param()
{
	SG_ADD(&degree, "degree", "degree of the kernel", MS_AVAILABLE);
	SG_ADD(&AA_matrix, "AA_matrix", "128*128 scalar product matrix", MS_NOT_AVAILABLE);
	SG_ADD(&width, "width", "width of Gaussian", MS_AVAILABLE);
	SG_ADD(&nof_sequences, "nof_sequences", "length of the sequence",
	    MS_NOT_AVAILABLE);
	m_parameters->add_vector(&sequences, &nof_sequences, "sequences", "the sequences as a part of profile");
	SG_ADD(&max_sequence_length,
	    "max_sequence_length", "max length of the sequence", MS_NOT_AVAILABLE);
}

void CSpectrumRBFKernel::register_alphabet()
{
	SG_ADD((CSGObject**)&alphabet, "alphabet", "the alphabet used by kernel",
	    MS_NOT_AVAILABLE);
}

void CSpectrumRBFKernel::init()
{
	alphabet = NULL;
	degree = 0;
	width = 0.0;
	sequences = NULL;
	string_features = NULL;
	nof_sequences = 0;
	max_sequence_length = 0;

	initialized = false;

	max_mismatch = 0;
	target_letter_0 = 0;
}
