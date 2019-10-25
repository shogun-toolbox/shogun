/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Viktor Gal, Soeren Sonnenburg,
 *          Weijie Lin, Bjoern Esser, Saurabh Goyal
 */

#include <vector>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Trie.h>
#include <shogun/base/Parallel.h>

#include <shogun/kernel/string/SpectrumRBFKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/mathematics/Math.h>

#include <vector>
#include <string>
#include <fstream>

#include <assert.h>

using namespace shogun;

SpectrumRBFKernel::SpectrumRBFKernel()
  : StringKernel<char>(0)
{
    init();
	register_param();
}

SpectrumRBFKernel::SpectrumRBFKernel (int32_t size, float64_t *AA_matrix_, int32_t degree_, float64_t width_)
  : StringKernel<char>(size), alphabet(NULL), degree(degree_), width(width_), sequences(NULL), string_features(NULL), nof_sequences(0), max_sequence_length(0)
{
	init();
	register_param();

	target_letter_0=-1 ;

	AA_matrix=SGMatrix<float64_t>(128,128);

	sg_memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;

	read_profiles_and_sequences();
	std::vector<SGVector<char>> string_list;
	string_list.reserve(nof_sequences);
	std::copy_n(sequences, nof_sequences, std::back_inserter(string_list));

	//string_features = new CStringFeatures<char>(sequences, nof_sequences, max_sequence_length, PROTEIN);
	string_features = std::make_shared<StringFeatures<char>>(string_list, IUPAC_AMINO_ACID);

	init(string_features, string_features);
}

SpectrumRBFKernel::SpectrumRBFKernel(
	std::shared_ptr<StringFeatures<char>> l, std::shared_ptr<StringFeatures<char>> r, int32_t size, float64_t* AA_matrix_, int32_t degree_, float64_t width_)
: StringKernel<char>(size), alphabet(NULL), degree(degree_), width(width_), sequences(NULL), string_features(NULL), nof_sequences(0), max_sequence_length(0)
{
	target_letter_0=-1 ;

	AA_matrix=SGMatrix<float64_t>(128,128);
	sg_memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;

	init(l, r);
	register_param();
}

SpectrumRBFKernel::~SpectrumRBFKernel()
{
	cleanup();

	SG_FREE(sequences);
}

void SpectrumRBFKernel::read_profiles_and_sequences()
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
	SG_DEBUG("initializing background")
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

	SG_DEBUG("Reading profiles from {}", filename)
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
						SG_DEBUG("Skipping aa in sequence {}", pos)
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
							//SG_DEBUG("seq {} aa {} value {} p {}  bg {}", i, j, value,p, background[j])
						}

						if (all_zeros)
						{
							SG_DEBUG(">>>>>>>>>>>>>>> all zeros")
							if (aa !="B" && aa != "X" && aa != "Z")
							{
								//profile[i][temp_profile_index]=-log(C+(1-C)*background[re_candidate[temp_profile_index]]);
								int32_t aa_index = aa_to_index[(int)aa.c_str()[0]];
								double value = -1* std::log(C+(1-C)*background[aa_index]); // taken from Leslie's example, C actually corresponds to 1/(1+C)
								SG_DEBUG("before {}", profiles.back()[(i-1) * 20 + aa_index])
								curr_profile[(i*20) + aa_index] = value;
								SG_DEBUG(">>> aa {} \t {} \t {}", aa.c_str()[0], aa_index, value)

								/*
								for (int z=0; z <20; ++z)
								{
									SG_DEBUG(" {} \t {}\t", z, curr_profile[z])
								}
								SG_DEBUG("")
								*/
							}
						}
					}
			}

			if (curr_profile.size() != 20 * sequence.length())
	    {
				error("Something's wrong with the profile.");
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
	sequences = SG_MALLOC(SGVector<char>, nof_sequences);

	int max_len = 0;
	for (int i=0; i < nof_sequences; ++i)
	{
		int len = seqs[i].length();
		sequences[i] = SGVector<char>(len+1);
		sequences[i].vlen = len;
		strcpy(sequences[i].vector, seqs[i].c_str());

		if (len > max_len) max_len = len;
	}

	max_sequence_length = max_len;
	//string_features = new CStringFeatures<char>(sequences, nof_sequences, max_sequence_length, PROTEIN);

}

bool SpectrumRBFKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
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

	StringKernel<char>::init(l,r);

	SG_DEBUG("lhs_changed: {}", lhs_changed)
	SG_DEBUG("rhs_changed: {}", rhs_changed)

	auto sf_l=std::static_pointer_cast<StringFeatures<char>>(l);
	auto sf_r=std::static_pointer_cast<StringFeatures<char>>(r);


	alphabet=sf_l->get_alphabet();
	auto ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties &= ((uint64_t) (-1)) ^ (KP_LINADD | KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())



	return init_normalizer();
}

void SpectrumRBFKernel::cleanup()
{


	alphabet=NULL;

	Kernel::cleanup();
}

inline bool isaa(char c)
{
  if (c<65 || c>89 || c=='B' || c=='J' || c=='O' || c=='U' || c=='X' || c=='Z')
    return false ;
  return true ;
}

float64_t SpectrumRBFKernel::AA_helper(const char* path, const int seq_degree, const char* joint_seq, unsigned int index)
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
	  if (Math::is_nan(diff))
	    fprintf(stderr, "nan occurred: '%c' '%c'\n", path[i], joint_seq[index+i]) ;
	}
    }

  return exp( - diff/width) ;
}

float64_t SpectrumRBFKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	char* avec = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, afree);
	char* bvec = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, bfree);

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
	  {
	    for (int32_t j=0; j<blen; j++)
	      {
		if ((i+degree<=alen) && (j+degree<=blen))
		  result += AA_helper(&(avec[i]), degree, bvec, j) ;
	      }
	  }

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, afree);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, bfree);
	return result;
}

bool SpectrumRBFKernel::set_AA_matrix(
	float64_t* AA_matrix_)
{

	if (AA_matrix_)
	{
		SG_DEBUG("Setting AA_matrix")
		sg_memcpy(AA_matrix.matrix, AA_matrix_, 128*128*sizeof(float64_t)) ;
		return true ;
	}

	return false;
}

void SpectrumRBFKernel::register_param()
{
	SG_ADD(&degree, "degree", "degree of the kernel", ParameterProperties::HYPER);
	SG_ADD(&AA_matrix, "AA_matrix", "128*128 scalar product matrix");
	SG_ADD(&width, "width", "width of Gaussian", ParameterProperties::HYPER);
	SG_ADD(&nof_sequences, "nof_sequences", "length of the sequence");

	/*m_parameters->add_vector(&sequences, &nof_sequences, "sequences", "the sequences as a part of profile");*/
	watch_param("sequences", &sequences, &nof_sequences);

	SG_ADD(&max_sequence_length,
	    "max_sequence_length", "max length of the sequence");
}

void SpectrumRBFKernel::register_alphabet()
{
	SG_ADD((std::shared_ptr<SGObject>*)&alphabet, "alphabet", "the alphabet used by kernel");
}

void SpectrumRBFKernel::init()
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
