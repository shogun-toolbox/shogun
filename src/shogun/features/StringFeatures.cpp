#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/MemoryMappedFile.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/Path.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <tchar.h>
#include <strsafe.h>
#include <vector>
#else
#include <unistd.h>
#include <algorithm>
#include <utility>

#endif

namespace shogun
{

template<class ST> StringFeatures<ST>::StringFeatures() : Features(0)
{
	init();
	alphabet=std::make_shared<Alphabet>();

}

template<class ST> StringFeatures<ST>::StringFeatures(EAlphabet alpha) : Features(0)
{
	init();

	alphabet=std::make_shared<Alphabet>(alpha);

	num_symbols=alphabet->get_num_symbols();
	original_num_symbols=num_symbols;
}

template<class ST> StringFeatures<ST>::StringFeatures(const std::vector<SGVector<ST>>& string_list, EAlphabet alpha)
: StringFeatures(alpha)
{
	set_features(string_list);
}

template<class ST> StringFeatures<ST>::StringFeatures(const std::vector<SGVector<ST>>& string_list, std::shared_ptr<Alphabet> alpha)
: StringFeatures(alpha)
{
	set_features(string_list);
}

template<class ST> StringFeatures<ST>::StringFeatures(std::shared_ptr<Alphabet> alpha)
: Features(0)
{
	ASSERT(alpha)
	init();

	alphabet=alpha;
	num_symbols=alphabet->get_num_symbols();
	original_num_symbols=num_symbols;
}

template<class ST> StringFeatures<ST>::StringFeatures(std::shared_ptr<File> loader, EAlphabet alpha)
: StringFeatures(alpha)
{
	load(loader);
}

template<class ST> StringFeatures<ST>::~StringFeatures()
{
	cleanup();
}

template<class ST> void StringFeatures<ST>::cleanup()
{
	remove_all_subsets();

	if (single_string.vector)
		single_string = SGVector<ST>();
	else
		cleanup_feature_vectors(0, get_num_vectors()-1);

	/*
	if (single_string)
	{
		SG_FREE(single_string);
		single_string=NULL;
	}
	else
		cleanup_feature_vectors(0, get_num_vectors()-1);
	*/

	features.clear();
	symbol_mask_table = SGVector<ST>();

	/* start with a fresh alphabet, but instead of emptying the histogram
	 * create a new object (to leave the alphabet object alone if it is used
	 * by others)
	 */
	auto alpha=std::make_shared<Alphabet>(alphabet->get_alphabet());

	alphabet=alpha;

}

template<class ST> void StringFeatures<ST>::cleanup_feature_vector(int32_t num)
{
	ASSERT(num<get_num_vectors())

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);
	features[real_num] = SGVector<ST>();
}

template<class ST> void StringFeatures<ST>::cleanup_feature_vectors(int32_t start, int32_t stop)
{
	if(get_num_vectors())
	{
		ASSERT(stop<get_num_vectors())
		ASSERT(stop >= start && start >= 0);

		for (int32_t i=start; i<=stop; i++)
		{
			int32_t real_num=m_subset_stack->subset_idx_conversion(i);
			features[real_num] = SGVector<ST>();
		}
	}
}

template<class ST> EFeatureClass StringFeatures<ST>::get_feature_class() const { return C_STRING; }

template<class ST> EFeatureType StringFeatures<ST>::get_feature_type() const { return F_UNKNOWN; }

template<class ST> std::shared_ptr<Alphabet> StringFeatures<ST>::get_alphabet() const
{

	return alphabet;
}

template<class ST> std::shared_ptr<Features> StringFeatures<ST>::duplicate() const
{
	return std::make_shared<StringFeatures>(*this);
}

template<class ST> SGVector<ST> StringFeatures<ST>::get_feature_vector(int32_t num)
{
	if (num>=get_num_vectors())
	{
		error("Index out of bounds (number of strings {}, you "
				"requested {})", get_num_vectors(), num);
	}

	int32_t l;
	bool free_vec;
	ST* vec=get_feature_vector(num, l, free_vec);
	ST* dst=SG_MALLOC(ST, l);
	sg_memcpy(dst, vec, l*sizeof(ST));
	free_feature_vector(vec, num, free_vec);
	return SGVector<ST>(dst, l, true);
}

template<class ST> void StringFeatures<ST>::set_feature_vector(SGVector<ST> vector, int32_t num)
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot set feature vector");

	if (num>=get_num_vectors())
	{
		error("Index out of bounds (number of strings {}, you "
				"requested {})", get_num_vectors(), num);
	}

	if (vector.vlen<=0)
		error("String has zero or negative length");

	features[num] = vector.clone();
}

template<class ST> void StringFeatures<ST>::enable_on_the_fly_preprocessing()
{
	preprocess_on_get=true;
}

template<class ST> void StringFeatures<ST>::disable_on_the_fly_preprocessing()
{
	preprocess_on_get=false;
}

template<class ST> ST* StringFeatures<ST>::get_feature_vector(int32_t num, int32_t& len, bool& dofree)
{
	if (num>=get_num_vectors())
		error("Requested feature vector with index {} while total num is {}", num, get_num_vectors());

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (!preprocess_on_get)
	{
		dofree=false;
		len=features[real_num].vlen;
		return features[real_num].vector;
	}
	else
	{
		SG_DEBUG("computing feature vector!")
		ST* feat=compute_feature_vector(num, len);
		dofree=true;

		if (get_num_preprocessors())
		{
			ST* tmp_feat_before=feat;

			for (int32_t i=0; i<get_num_preprocessors(); i++)
			{
				auto p=std::dynamic_pointer_cast<StringPreprocessor<ST>>(get_preprocessor(i));
				feat=p->apply_to_string(tmp_feat_before, len);
				SG_FREE(tmp_feat_before);
				tmp_feat_before=feat;
			}
		}
		// TODO: implement caching
		return feat;
	}
}

template<class ST> std::shared_ptr<StringFeatures<ST>> StringFeatures<ST>::get_transposed()
{
	return std::make_shared<StringFeatures<ST>>(get_transposed_matrix(), alphabet);
}

template<class ST> std::vector<SGVector<ST>> StringFeatures<ST>::get_transposed_matrix()
{
	int32_t num_feat=get_num_vectors();
	int32_t num_vec=get_max_vector_length();
	ASSERT(have_same_length())

	SG_DEBUG("Allocating memory for transposed string features of size {}",
			int64_t(num_feat)*num_vec);

	std::vector<SGVector<ST>> sf;
	sf.reserve(num_vec);

	for (int32_t i=0; i<num_vec; i++)
	{
		sf.emplace_back(num_feat);
	}

	for (int32_t i=0; i<num_feat; i++)
	{
		int32_t len=0;
		bool free_vec=false;
		ST* vec=get_feature_vector(i, len, free_vec);

		for (int32_t j=0; j<num_vec; j++)
			sf[j].vector[i]=vec[j];

		free_feature_vector(vec, i, free_vec);
	}
	return sf;
}

template<class ST> void StringFeatures<ST>::free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
{
	if (num>=get_num_vectors())
	{
		error(
			"Trying to access string[{}] but num_str={}", num,
			get_num_vectors());
	}

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (feature_cache)
		feature_cache->unlock_entry(real_num);

	if (dofree)
		SG_FREE(feat_vec);
}

template<class ST> void StringFeatures<ST>::free_feature_vector(SGVector<ST> feat_vec, int32_t num)
{
	if (num>=get_num_vectors())
	{
		error(
			"Trying to access string[{}] but num_str={}", num,
			get_num_vectors());
	}

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (feature_cache)
		feature_cache->unlock_entry(real_num);
}

template<class ST> ST StringFeatures<ST>::get_feature(int32_t vec_num, int32_t feat_num)
{
	ASSERT(vec_num<get_num_vectors())

	int32_t len;
	bool free_vec;
	ST* vec=get_feature_vector(vec_num, len, free_vec);
	ASSERT(feat_num<len)
	ST result=vec[feat_num];
	free_feature_vector(vec, vec_num, free_vec);

	return result;
}

template<class ST> int32_t StringFeatures<ST>::get_vector_length(int32_t vec_num)
{
	ASSERT(vec_num<get_num_vectors())

	int32_t len;
	bool free_vec;
	ST* vec=get_feature_vector(vec_num, len, free_vec);
	free_feature_vector(vec, vec_num, free_vec);
	return len;
}

template<class ST> int32_t StringFeatures<ST>::get_max_vector_length() const
{
	int32_t max_string_length=0;
	index_t num_str=get_num_vectors();

	for (int32_t i=0; i<num_str; i++)
	{
		max_string_length=Math::max(max_string_length,
			features[m_subset_stack->subset_idx_conversion(i)].vlen);
	}
	return max_string_length;
}

template<class ST> int32_t StringFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : features.size();
}

template<class ST> floatmax_t StringFeatures<ST>::get_num_symbols() { return num_symbols; }

template<class ST> floatmax_t StringFeatures<ST>::get_max_num_symbols() { return Math::powl(2,sizeof(ST)*8); }

template<class ST> floatmax_t StringFeatures<ST>::get_original_num_symbols() { return original_num_symbols; }

template<class ST> int32_t StringFeatures<ST>::get_order() { return order; }

template<class ST> ST StringFeatures<ST>::get_masked_symbols(ST symbol, uint8_t mask)
{
	ASSERT(symbol_mask_table)
	return symbol_mask_table[mask] & symbol;
}

template<class ST> ST StringFeatures<ST>::shift_offset(ST offset, int32_t amount)
{
	ASSERT(alphabet)
	return (offset << (amount*alphabet->get_num_bits()));
}

template<class ST> ST StringFeatures<ST>::shift_symbol(ST symbol, int32_t amount)
{
	ASSERT(alphabet)
	return (symbol >> (amount*alphabet->get_num_bits()));
}

template<class ST> void StringFeatures<ST>::load_ascii_file(char* fname, bool remap_to_bin,
		EAlphabet ascii_alphabet, EAlphabet binary_alphabet)
{
	remove_all_subsets();

	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	uint8_t* dummy=SG_MALLOC(uint8_t, blocksize);
	uint8_t* overflow=NULL;
	int32_t overflow_len=0;

	cleanup();

	auto alpha=std::make_shared<Alphabet>(ascii_alphabet);
	auto alpha_bin=std::make_shared<Alphabet>(binary_alphabet);

	FILE* f=fopen(fname, "ro");
	int32_t num_vectors = 0;
	if (f)
	{
		io::info("counting line numbers in file {}", fname);
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(f, 0, SEEK_END);
		size_t fsize=ftell(f);
		rewind(f);

		if (blocksize>fsize)
			blocksize=fsize;

		SG_DEBUG("block_size={} file_size={}", blocksize, fsize)

		auto pb = SG_PROGRESS(range(fsize));
		size_t sz=blocksize;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, f);
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_vectors++;
					required_blocksize=Math::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			pb.print_progress();
		}
		pb.complete();

		io::info("found {} strings", num_vectors);
		SG_FREE(dummy);
		blocksize=required_blocksize;
		dummy=SG_MALLOC(uint8_t, blocksize);
		overflow=SG_MALLOC(uint8_t, blocksize);
		features.clear();
		features.resize(num_vectors);

		auto pb2 =
			PRange<int>(range(num_vectors), "LOADING: ", UTF8, []() {
				return true;
			});
		rewind(f);
		sz=blocksize;
		int32_t lines=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, f);

			size_t old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					int32_t len=i-old_sz;
					//io::print("i:%d len:{} old_sz:{}\n", i, len, old_sz)

					features[lines] = SGVector<ST>(len);
					if (remap_to_bin)
					{
						for (int32_t j=0; j<overflow_len; j++)
							features[lines].vector[j]=alpha->remap_to_bin(overflow[j]);
						for (int32_t j=0; j<len; j++)
							features[lines].vector[j+overflow_len]=alpha->remap_to_bin(dummy[old_sz+j]);
						alpha->add_string_to_histogram(&dummy[old_sz], len);
						alpha_bin->add_string_to_histogram(features[lines].vector, features[lines].vlen);
					}
					else
					{
						for (int32_t j=0; j<overflow_len; j++)
							features[lines].vector[j]=overflow[j];
						for (int32_t j=0; j<len; j++)
							features[lines].vector[j+overflow_len]=dummy[old_sz+j];
						alpha->add_string_to_histogram(&dummy[old_sz], len);
						alpha->add_string_to_histogram(features[lines].vector, features[lines].vlen);
					}

					// clear overflow
					overflow_len=0;

					//Math::display_vector(features[lines].string, len);
					old_sz=i+1;
					lines++;
					pb2.print_progress();
				}
			}
			pb2.complete();

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}

		if (alpha->check_alphabet_size() && alpha->check_alphabet())
		{
			io::info("file successfully read");
			io::info("max_string_length={}", get_max_vector_length());
			io::info("num_strings={}", get_num_vectors());
		}
		fclose(f);
	}

	SG_FREE(dummy);
	SG_FREE(overflow);

	if (remap_to_bin)
	{
		alphabet=alpha_bin;
	}
	else
	{
		alphabet=alpha;
	}

	num_symbols=alphabet->get_num_symbols();
}

template<class ST> bool StringFeatures<ST>::load_fasta_file(const char* fname, bool ignore_invalid)
{
	remove_all_subsets();

	int32_t i=0;
	uint64_t len=0;
	uint64_t offs=0;
	int32_t num=0;

	MemoryMappedFile<char> f(fname);

	while (true)
	{
		char* s=f.get_line(len, offs);
		if (!s)
			break;

		if (len>0 && s[0]=='>')
			num++;
	}

	if (num==0)
		error("No fasta hunks (lines starting with '>') found");

	cleanup();

	alphabet=std::make_shared<Alphabet>(DNA);
	num_symbols=alphabet->get_num_symbols();

	std::vector<SGVector<ST>> strings;
	strings.reserve(num);
	offs=0;

	for (i=0;i<num; i++)
	{
		uint64_t id_len=0;
		char* id=f.get_line(id_len, offs);

		char* fasta=f.get_line(len, offs);
		char* s=fasta;
		int32_t fasta_len=0;
		int32_t spanned_lines=0;

		while (true)
		{
			if (!s || len==0)
				error("Error reading fasta entry in line {} len={}", 4*i+1, len);

			if (s[0]=='>' || offs==f.get_size())
			{
				offs-=len+1; // seek to beginning
				if (offs==f.get_size())
				{
					SG_DEBUG("at EOF")
					fasta_len+=len;
				}

				len=fasta_len-spanned_lines;
				strings.emplace_back(len);

				ST* str=strings.back().vector;
				int32_t idx=0;
				SG_DEBUG("'{:{}}', len={}, spanned_lines={}", id, (int32_t) id_len, (int32_t) len, (int32_t) spanned_lines)

				for (int32_t j=0; j<fasta_len; j++)
				{
					if (fasta[j]=='\n')
						continue;

					ST c=(ST) fasta[j];

					if (ignore_invalid  && !alphabet->is_valid((uint8_t) fasta[j]))
						c=(ST) 'A';

					if (uint64_t(idx)>=len)
						error("idx={} j={} fasta_len={}, spanned_lines={} str='{:{}}'", idx, j, fasta_len, spanned_lines, (char*)str, idx);
					str[idx++]=c;
				}

				break;
			}

			spanned_lines++;
			fasta_len+=len+1; // including '\n'
			s=f.get_line(len, offs);
		}
	}
	return set_features(strings);
}

template<class ST> bool StringFeatures<ST>::load_fastq_file(const char* fname,
		bool ignore_invalid, bool bitremap_in_single_string)
{
	remove_all_subsets();

	MemoryMappedFile<char> f(fname);

	int32_t i=0;
	uint64_t len=0;
	uint64_t offs=0;

	int32_t num=f.get_num_lines();
	int32_t max_len=0;

	if (num%4)
		error("Number of lines must be divisible by 4 in fastq files");
	num/=4;

	cleanup();

	alphabet=std::make_shared<Alphabet>(DNA);

	std::vector<SGVector<ST>> strings;

	ST* str=NULL;
	if (bitremap_in_single_string)
	{
		strings.reserve(1);
		strings.emplace_back(num);
		f.get_line(len, offs);
		f.get_line(len, offs);
		order=len;
		max_len=num;
		offs=0;
		original_num_symbols=alphabet->get_num_symbols();
		str=SG_MALLOC(ST, len);
	}
	else
		strings.resize(num);

	for (i=0;i<num; i++)
	{
		if (!f.get_line(len, offs))
			error("Error reading 'read' identifier in line {}", 4*i);

		char* s=f.get_line(len, offs);
		if (!s || len==0)
			error("Error reading 'read' in line {} len={}", 4*i+1, len);

		if (bitremap_in_single_string)
		{
			if (len!=(uint64_t) order)
				error("read in line {} not of length {} (is {})", 4*i+1, order, len);
			for (int32_t j=0; j<order; j++)
				str[j]=(ST) alphabet->remap_to_bin((uint8_t) s[j]);

			strings[0].vector[i]=embed_word(str, order);
		}
		else
		{
			strings[i] = SGVector<ST>(len);
			str=strings[i].vector;

			if (ignore_invalid)
			{
				for (uint64_t j=0; j<len; j++)
				{
					if (alphabet->is_valid((uint8_t) s[j]))
						str[j]= (ST) s[j];
					else
						str[j]= (ST) 'A';
				}
			}
			else
			{
				for (uint64_t j=0; j<len; j++)
					str[j]= (ST) s[j];
			}
			max_len=Math::max(max_len, (int32_t) len);
		}


		if (!f.get_line(len, offs))
			error("Error reading 'read' quality identifier in line {}", 4*i+2);

		if (!f.get_line(len, offs))
			error("Error reading 'read' quality in line {}", 4*i+3);
	}

	if (bitremap_in_single_string)
		num=1;

	features=std::move(strings);

	return true;
}

template<class ST> bool StringFeatures<ST>::load_from_directory(char* dirname)
{
	remove_all_subsets();

	std::vector<std::string> children;
	auto fs_registry = env();
	require(!fs_registry->is_directory(dirname),
		"Specified path ('{}') is not a directory!", dirname);
	auto r = fs_registry->get_children(dirname, &children);
	if (r)
		throw io::to_system_error(r);

	if (children.size() <= 0)
	{
		error("error calling scandir - no files found");
		return false;
	}
	else
	{
		int32_t num=0;
		int64_t max_buffer_size = -1;

		//usually n==num_vec, but it might not in race conditions
		//(file perms modified, file erased)
		std::vector<SGVector<ST>> strings;
		strings.reserve(children.size());
		std::string buffer;
		for (auto v: children)
		{
			auto fname = io::join_path(dirname, v);

			if (fs_registry->is_directory(fname))
			{
				int64_t filesize = fs_registry->get_file_size(fname);

				std::unique_ptr<io::RandomAccessFile> file;
				if (!fs_registry->new_random_access_file(fname, &file))
				{
					SG_DEBUG("{}:{}", fname.c_str(), filesize);
					std::string_view result;
					buffer.clear();
					if (max_buffer_size < filesize)
					{
						buffer.resize(filesize);
						max_buffer_size = filesize;
					}
					if (file->read(0, filesize, &result, &(buffer[0])))
						error("failed to read file");
					int64_t sg_string_len = filesize/(int64_t)sizeof(ST);
					strings.emplace_back(sg_string_len);
					sg_memcpy(const_cast<char*>(result.data()), strings[num].vector, filesize);
					++num;
				}
			}
			else
				SG_DEBUG("Skipping {} as it's a directory", fname.c_str());
		}

		if (num>0)
		{
			return set_features(strings);
		}

	}
	return false;
}

template<class ST> bool StringFeatures<ST>::set_features(const std::vector<SGVector<ST>>& string_list)
{
	return set_features(string_list.data(), string_list.size());
}

template<class ST> bool StringFeatures<ST>::set_features(const SGVector<ST>* p_features, int32_t p_num_vectors)
{
	if (m_subset_stack->has_subsets())
		error("Cannot call set_features() with subset.");

	if (p_features)
	{
		auto alpha=std::make_shared<Alphabet>(alphabet->get_alphabet());

		//compute histogram for char/byte
		for (int32_t i=0; i<p_num_vectors; i++)
			alpha->add_string_to_histogram( p_features[i].vector, p_features[i].vlen);

		io::info("max_value_in_histogram:{}", alpha->get_max_value_in_histogram());
		io::info("num_symbols_in_histogram:{}", alpha->get_num_symbols_in_histogram());

		if (alpha->check_alphabet_size() && alpha->check_alphabet())
		{
			cleanup();


			alphabet=alpha;


			// TODO remove copying
			features.clear();
			features.reserve(p_num_vectors);
			std::copy_n(p_features, p_num_vectors, std::back_inserter(features));

			return true;
		}
	}

	return false;
}

template<class ST> bool StringFeatures<ST>::append_features(std::shared_ptr<StringFeatures<ST>> sf)
{
	ASSERT(sf)

	if (m_subset_stack->has_subsets())
		error("Cannot call set_features() with subset.");

	std::vector<SGVector<ST>> new_features;
	new_features.reserve(sf->get_num_vectors());

	index_t sf_num_str=sf->get_num_vectors();
	for (int32_t i=0; i<sf_num_str; i++)
	{
		int32_t real_i = sf->m_subset_stack->subset_idx_conversion(i);
		new_features[i] = sf->features[real_i].clone();
	}
	return append_features(new_features);
}

template<class ST> bool StringFeatures<ST>::append_features(const std::vector<SGVector<ST>>& p_features)
{
	if (m_subset_stack->has_subsets())
		error("Cannot call set_features() with subset.");

	if (features.empty())
		return set_features(p_features);

	auto alpha=std::make_shared<Alphabet>(alphabet->get_alphabet());

	//compute histogram for char/byte
	for (int32_t i=0; i<p_features.size(); i++)
		alpha->add_string_to_histogram( p_features[i].vector, p_features[i].vlen);

	io::info("max_value_in_histogram:{}", alpha->get_max_value_in_histogram());
	io::info("num_symbols_in_histogram:{}", alpha->get_num_symbols_in_histogram());

	if (alpha->check_alphabet_size() && alpha->check_alphabet())
	{
		for (int32_t i=0; i<p_features.size(); i++)
			alphabet->add_string_to_histogram(p_features[i].vector, p_features[i].vlen);

		int32_t old_num_vectors=get_num_vectors();
		int32_t num_vectors=old_num_vectors+p_features.size();
		std::vector<SGVector<ST>> new_features;
		new_features.reserve(num_vectors);

		for (int32_t i=0; i<get_num_vectors(); i++)
		{
			if (i<old_num_vectors)
				new_features.push_back(features[i]);
			else
				new_features.push_back(p_features[i-old_num_vectors]);
		}

		this->features=std::move(new_features);

		return true;
	}


	return false;
}

template<class ST> const std::vector<SGVector<ST>>& StringFeatures<ST>::get_string_list() const
{
	if (m_subset_stack->has_subsets())
		error("get features() is not possible on subset");

	return features;
}

template<class ST> std::vector<SGVector<ST>>& StringFeatures<ST>::get_string_list()
{
	return const_cast<std::vector<SGVector<ST>>&>(
		std::as_const(*this).get_string_list());
}

template<class ST> std::vector<SGVector<ST>> StringFeatures<ST>::copy_features()
{
	ASSERT(get_num_vectors()>0)

	std::vector<SGVector<ST>> new_feat;
	new_feat.reserve(get_num_vectors());

	for (int32_t i=0; i<get_num_vectors(); i++)
	{
		SGVector<ST> vec =get_feature_vector(i);
		new_feat.push_back(vec.clone());
		free_feature_vector(vec, i);
	}

	return new_feat;
}

template<class ST> void StringFeatures<ST>::get_features(std::vector<SGVector<ST>>* dst)
{
	*dst=copy_features();
}

template<class ST> bool StringFeatures<ST>::load_compressed(char* src, bool decompress)
{
	remove_all_subsets();

	FILE* file=NULL;
	int32_t num_vectors = 0;
	int32_t max_string_length = 0;

	if (!(file=fopen(src, "r")))
		return false;
	cleanup();

	// header shogun v0
	char id[4];
	if (fread(&id[0], sizeof(char), 1, file)!=1)
		error("failed to read header");
	ASSERT(id[0]=='S')
	if (fread(&id[1], sizeof(char), 1, file)!=1)
		error("failed to read header");
	ASSERT(id[1]=='G')
	if (fread(&id[2], sizeof(char), 1, file)!=1)
		error("failed to read header");
	ASSERT(id[2]=='V')
	if (fread(&id[3], sizeof(char), 1, file)!=1)
		error("failed to read header");
	ASSERT(id[3]=='0')

	//compression type
	uint8_t c;
	if (fread(&c, sizeof(uint8_t), 1, file)!=1)
		error("failed to read compression type");
	auto compressor= std::make_shared<Compressor>((E_COMPRESSION_TYPE) c);
	//alphabet
	uint8_t a;
	alphabet.reset();
	if (fread(&a, sizeof(uint8_t), 1, file)!=1)
		error("failed to read compression alphabet");
	alphabet=std::make_shared<Alphabet>((EAlphabet) a);
	// number of vectors
	if (fread(&num_vectors, sizeof(int32_t), 1, file)!=1)
		error("failed to read compression number of vectors");
	ASSERT(num_vectors>0)
	// maximum string length
	if (fread(&max_string_length, sizeof(int32_t), 1, file)!=1)
		error("failed to read maximum string length");
	ASSERT(max_string_length>0)

	features.clear();
	features.reserve(num_vectors);
	// vectors
	for (int32_t i=0; i<num_vectors; i++)
	{
		// vector len compressed
		int32_t len_compressed;
		if (fread(&len_compressed, sizeof(int32_t), 1, file)!=1)
			error("failed to read vector length compressed");
		// vector len uncompressed
		int32_t len_uncompressed;
		if (fread(&len_uncompressed, sizeof(int32_t), 1, file)!=1)
			error("failed to read vector length uncompressed");

		// vector raw data
		if (decompress)
		{
			features.emplace_back(len_uncompressed);
			uint8_t* compressed=SG_MALLOC(uint8_t, len_compressed);
			if (fread(compressed, sizeof(uint8_t), len_compressed, file)!=(size_t) len_compressed)
				error("failed to read compressed data (expected {} bytes)", len_compressed);
			uint64_t uncompressed_size=len_uncompressed;
			uncompressed_size*=sizeof(ST);
			compressor->decompress(compressed, len_compressed,
					(uint8_t*) features[i].vector, uncompressed_size);
			SG_FREE(compressed);
			ASSERT(uncompressed_size==((uint64_t) len_uncompressed)*sizeof(ST))
		}
		else
		{
			int32_t offs = std::ceil(2.0 * sizeof(int32_t) / sizeof(ST));
			features.emplace_back(len_compressed+offs);
			int32_t* feat32ptr=((int32_t*) (features[i].vector));
			feat32ptr[0]=(int32_t) len_compressed;
			feat32ptr[1]=(int32_t) len_uncompressed;
			uint8_t* compressed=(uint8_t*) (&features[i].vector[offs]);
			if (fread(compressed, 1, len_compressed, file)!=(size_t) len_compressed)
				error("failed to read uncompressed data");
		}
	}

	fclose(file);

	return false;
}

template<class ST> bool StringFeatures<ST>::save_compressed(char* dest, E_COMPRESSION_TYPE compression, int level)
{
	int32_t num_vectors = get_num_vectors();
	int32_t max_string_length = get_max_vector_length();

	if (m_subset_stack->has_subsets())
		error("save_compressed() is not possible on subset");

	FILE* file=NULL;

	if (!(file=fopen(dest, "wb")))
		return false;

	auto compressor= std::make_shared<Compressor>(compression);

	// header shogun v0
	const char* id="SGV0";
	fwrite(&id[0], sizeof(char), 1, file);
	fwrite(&id[1], sizeof(char), 1, file);
	fwrite(&id[2], sizeof(char), 1, file);
	fwrite(&id[3], sizeof(char), 1, file);

	//compression type
	uint8_t c=(uint8_t) compression;
	fwrite(&c, sizeof(uint8_t), 1, file);
	//alphabet
	uint8_t a=(uint8_t) alphabet->get_alphabet();
	fwrite(&a, sizeof(uint8_t), 1, file);
	// number of vectors
	fwrite(&num_vectors, sizeof(int32_t), 1, file);
	// maximum string length
	fwrite(&max_string_length, sizeof(int32_t), 1, file);

	// vectors
	for (int32_t i=0; i<num_vectors; i++)
	{
		int32_t len=-1;
		bool vfree;
		ST* vec=get_feature_vector(i, len, vfree);

		uint8_t* compressed=NULL;
		uint64_t compressed_size=0;

		compressor->compress((uint8_t*) vec, ((uint64_t) len)*sizeof(ST),
				compressed, compressed_size, level);

		int32_t len_compressed=(int32_t) compressed_size;
		// vector len compressed in bytes
		fwrite(&len_compressed, sizeof(int32_t), 1, file);
		// vector len uncompressed in number of elements of type ST
		fwrite(&len, sizeof(int32_t), 1, file);
		// vector raw data
		fwrite(compressed, compressed_size, 1, file);
		SG_FREE(compressed);

		free_feature_vector(vec, i, vfree);
	}

	fclose(file);
	return true;
}

template<class ST> int32_t StringFeatures<ST>::obtain_by_sliding_window(int32_t window_size, int32_t step_size, int32_t skip)
{
	if (m_subset_stack->has_subsets())
		not_implemented(SOURCE_LOCATION);

	int32_t num_vectors = get_num_vectors();
	int32_t max_string_length = get_max_vector_length();

	ASSERT(step_size>0)
	ASSERT(window_size>0)
	ASSERT(num_vectors==1 || single_string.vector)
	ASSERT(max_string_length>=window_size ||
			(single_string.vector && single_string.vlen>=window_size));

	//in case we are dealing with a single remapped string
	//allow remapping
	if (single_string.vector)
		num_vectors= (single_string.vlen-window_size)/step_size + 1;
	else if (num_vectors==1)
		num_vectors= (max_string_length-window_size)/step_size + 1;

	std::vector<SGVector<ST>> f;
	f.reserve(get_num_vectors());
	int32_t offs=0;
	for (int32_t i=0; i<get_num_vectors(); i++)
	{
		index_t l = offs+skip;
		index_t h = std::min(offs+window_size, features[0].size());
		f.push_back(features[0].slice(l, h));
		offs+=step_size;
	}
	single_string=features[0];
	features=std::move(f);

	return num_vectors;
}

template<class ST> int32_t StringFeatures<ST>::obtain_by_position_list(int32_t window_size, std::shared_ptr<DynamicArray<int32_t>> positions,
		int32_t skip)
{
	if (m_subset_stack->has_subsets())
		not_implemented(SOURCE_LOCATION);

	int32_t num_vectors = get_num_vectors();
	int32_t max_string_length = get_max_vector_length();

	ASSERT(positions)
	ASSERT(window_size>0)
	ASSERT(num_vectors==1 || single_string.vector)
	ASSERT(max_string_length>=window_size ||
			(single_string.vector && single_string.vlen>=window_size));

	num_vectors= positions->get_num_elements();
	ASSERT(num_vectors>0)

	int32_t len;

	//in case we are dealing with a single remapped string
	//allow remapping
	if (single_string.vector)
		len=single_string.vlen;
	else
	{
		single_string=features[0];
		len=max_string_length;
	}

	std::vector<SGVector<ST>> f;
	f.reserve(num_vectors);
	for (int32_t i=0; i<num_vectors; i++)
	{
		int32_t p=positions->get_element(i);

		if (p>=0 && p<=len-window_size)
		{
			index_t l = p+skip;
			index_t h = std::min(p+window_size, features[0].size());
			f.push_back(features[0].slice(l, h));
			f.back().vlen = window_size-skip;
		}
		else
		{
			num_vectors=1;
			features[0].vlen=len;
			single_string=SGVector<ST>();
			error("window (size:{}) starting at position[{}]={} does not fit in sequence(len:{})",
					window_size, i, p, len);
			return -1;
		}
	}

	features=std::move(f);
	return num_vectors;
}

template<class ST> bool StringFeatures<ST>::obtain_from_char(std::shared_ptr<StringFeatures<char>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return obtain_from_char_features(sf, start, p_order, gap, rev);
}

template<class ST> bool StringFeatures<ST>::have_same_length(int32_t len)
{
	int32_t max_string_length = get_max_vector_length();
	if (len!=-1)
	{
		if (len!=max_string_length)
			return false;
	}
	len=max_string_length;

	index_t num_str=get_num_vectors();
	for (int32_t i=0; i<num_str; i++)
	{
		if (get_vector_length(i)!=len)
			return false;
	}

	return true;
}

template<class ST> void StringFeatures<ST>::embed_features(int32_t p_order)
{
	if (m_subset_stack->has_subsets())
		not_implemented(SOURCE_LOCATION);

	ASSERT(alphabet->get_num_symbols_in_histogram() > 0)

	order=p_order;
	original_num_symbols=alphabet->get_num_symbols();
	int32_t max_val=alphabet->get_num_bits();

	if (p_order>1)
		num_symbols=Math::powl((floatmax_t) 2, (floatmax_t) max_val*p_order);
	else
		num_symbols=original_num_symbols;

	io::info("max_val (bit): {} order: {} -> results in num_symbols: {:.0f}", max_val, p_order, num_symbols);

	if ( ((floatmax_t) num_symbols) > Math::powl(((floatmax_t) 2),((floatmax_t) sizeof(ST)*8)) )
		io::warn("symbols did not fit into datatype \"{}\" ({})", (char) max_val, (int) max_val);

	ST mask=0;
	for (int32_t i=0; i<p_order*max_val; i++)
		mask= (mask<<1) | ((ST) 1);

	for (int32_t i=0; i<get_num_vectors(); i++)
	{
		int32_t len=features[i].vlen;

		if (len < p_order)
			error("Sequence must be longer than order ({} vs. {})", len, p_order);

		ST* str=features[i].vector;

		// convert first word
		for (int32_t j=0; j<p_order; j++)
			str[j]=(ST) alphabet->remap_to_bin(str[j]);
		str[0]=embed_word(&str[0], p_order);

		// convert the rest
		int32_t idx=0;
		for (int32_t j=p_order; j<len; j++)
		{
			str[j]=(ST) alphabet->remap_to_bin(str[j]);
			str[idx+1]= ((str[idx]<<max_val) | str[j]) & mask;
			idx++;
		}

		features[i].vlen=len-p_order+1;
	}

	compute_symbol_mask_table(max_val);
}

template<class ST> void StringFeatures<ST>::compute_symbol_mask_table(int64_t max_val)
{
	if (m_subset_stack->has_subsets())
		not_implemented(SOURCE_LOCATION);

	symbol_mask_table = SGVector<ST>(256);

	uint64_t mask=0;
	for (int32_t i=0; i< (int64_t) max_val; i++)
		mask=(mask<<1) | 1;

	for (int32_t i=0; i<256; i++)
	{
		uint8_t bits=(uint8_t) i;
		symbol_mask_table[i]=0;

		for (int32_t j=0; j<8; j++)
		{
			if (bits & 1)
				symbol_mask_table[i]|=mask<<(max_val*j);

			bits>>=1;
		}
	}
}

template<class ST> void StringFeatures<ST>::unembed_word(ST word, uint8_t* seq, int32_t len)
{
	uint32_t nbits= (uint32_t) alphabet->get_num_bits();

	ST mask=0;
	for (uint32_t i=0; i<nbits; i++)
		mask=(mask<<1) | (ST) 1;

	for (int32_t i=0; i<len; i++)
	{
		ST w=(word & mask);
		seq[len-i-1]=alphabet->remap_to_char((uint8_t) w);
		word>>=nbits;
	}
}

template<class ST> ST StringFeatures<ST>::embed_word(ST* seq, int32_t len)
{
	ST value=(ST) 0;
	uint32_t nbits= (uint32_t) alphabet->get_num_bits();
	for (int32_t i=0; i<len; i++)
	{
		value<<=nbits;
		value|=seq[i];
	}

	return value;
}

template<class ST> ST* StringFeatures<ST>::get_zero_terminated_string_copy(SGVector<ST> str)
{
	int32_t l=str.vlen;
	ST* s=SG_MALLOC(ST, l+1);
	sg_memcpy(s, str.vector, sizeof(ST)*l);
	s[l]='\0';
	return s;
}

template<class ST> void StringFeatures<ST>::set_feature_vector(int32_t num, ST* string, int32_t len)
{
	ASSERT(num<get_num_vectors())

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	features[real_num] = SGVector<ST>(string, len);
}

template<class ST> void StringFeatures<ST>::get_histogram(float64_t** hist, int32_t* rows, int32_t* cols, bool normalize)
{
	int32_t nsym=get_num_symbols();
	int32_t slen=get_max_vector_length();
	int64_t sz=int64_t(nsym)*slen*sizeof(float64_t);
	float64_t* h= SG_MALLOC(float64_t, sz);
	memset(h, 0, sz);

	float64_t* h_normalizer=SG_MALLOC(float64_t, slen);
	memset(h_normalizer, 0, slen*sizeof(float64_t));
	int32_t num_str=get_num_vectors();
	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len;
		bool free_vec;
		ST* vec=get_feature_vector(i, len, free_vec);
		for (int32_t j=0; j<len; j++)
		{
			h[int64_t(j)*nsym+alphabet->remap_to_bin(vec[j])]++;
			h_normalizer[j]++;
		}
		free_feature_vector(vec, i, free_vec);
	}

	if (normalize)
	{
		for (int32_t i=0; i<slen; i++)
		{
			for (int32_t j=0; j<nsym; j++)
			{
				if (h_normalizer && h_normalizer[i])
					h[int64_t(i)*nsym+j]/=h_normalizer[i];
			}
		}
	}
	SG_FREE(h_normalizer);

	*hist=h;
	*rows=nsym;
	*cols=slen;
}

template<class ST> 
void StringFeatures<ST>::create_random(float64_t* hist, int32_t rows, int32_t cols, int32_t num_vec, int32_t seed)
{
	std::mt19937_64 prng(seed);
	ASSERT(rows == get_num_symbols())
	cleanup();
	float64_t* randoms=SG_MALLOC(float64_t, cols);
	std::vector<SGVector<ST>> sf;
	sf.reserve(num_vec);

	for (int32_t i=0; i<num_vec; i++)
	{
		sf.emplace_back(cols);

		random::fill_array(randoms, randoms + cols, 0.0, 1.0, prng);

		for (int32_t j=0; j<cols; j++)
		{
			float64_t lik=hist[int64_t(j)*rows+0];

			int32_t c;
			for (c=0; c<rows-1; c++)
			{
				if (randoms[j]<=lik)
					break;
				lik+=hist[int64_t(j)*rows+c+1];
			}
			sf[i].vector[j]=alphabet->remap_to_char(c);
		}
	}
	SG_FREE(randoms);
	set_features(sf);
}

/*
CStringFeatures<SSKTripleFeature>* obtain_sssk_triple_from_cha(int d1, int d2)
{
	int *s;
	int32_t nStr=get_num_vectors();

	int32_t nfeat=0;
	for (int32_t i=0; i < nStr; ++i)
		nfeat += get_vector_length[i] - d1 -d2;
	SGVector<SSKFeature>* F= SG_MALLOC(SGVector<SSKFeature>, nfeat);
	int32_t c=0;
	for (int32_t i=0; i < nStr; ++i)
	{
	int32_t len;
	bool free_vec;
	ST* S=get_feature_vector(vec_num, len, free_vec);
	free_feature_vector(vec, vec_num, free_vec);
		int32_t n=len - d1 - d2;
		s=S[i];
		for (int32_t j=0; j < n; ++j)
		{
			F[c].feature1=s[j];
			F[c].feature2=s[j+d1];
			F[c].feature3=s[j+d1+d2];
			F[c].group=i;
			c++;
		}
	}
	ASSERT(nfeat==c)
	return F;
}

CStringFeatures<SSKFeature>* obtain_sssk_double_from_char(int **S, int *len, int nStr, int d1)
{
	int i, j;
	int n, nfeat;
	int *group;
	int *features;
	int *s;
	int c;
	SSKFeatures *F;

	nfeat=0;
	for (i=0; i < nStr; ++i)
		nfeat += len[i] - d1;
	group=(int *)SG_MALLOC(nfeat*sizeof(int));
	features=(int *)SG_MALLOC(nfeat*2*sizeof(int *));
	c=0;
	for (i=0; i < nStr; ++i)
	{
		n=len[i] - d1;
		s=S[i];
		for (j=0; j < n; ++j)
		{
			features[c]=s[j];
			features[c+nfeat]=s[j+d1];
			group[c]=i;
			c++;
		}
	}
	if (nfeat!=c)
		printf("Something is wrong...\n");
	F=(SSKFeatures *)SG_MALLOC(sizeof(SSKFeatures));
	(*F).features=features;
	(*F).group=group;
	(*F).n=nfeat;
	return F;
}
*/

template<class ST> std::shared_ptr<Features> StringFeatures<ST>::copy_subset(
		SGVector<index_t> indices) const
{
	/* string list to create new CStringFeatures from */
	std::vector<SGVector<ST>> list_copy(indices.vlen);

	/* copy all features */
	for (index_t i=0; i<indices.vlen; ++i)
	{
		/* index with respect to possible subset */
		index_t real_idx=m_subset_stack->subset_idx_conversion(indices.vector[i]);

		/* copy string */
		SGVector<ST> current_string=features[real_idx];
		SGVector<ST> string_copy = current_string.clone();
		list_copy[i]=string_copy;
	}

	/* create copy instance */
	auto result=std::make_shared<StringFeatures<ST>>();

	/* keep things from original features (otherwise assertions in x-val) */
	result->order=order;
	result->compute_symbol_mask_table(result->alphabet->get_num_symbols());

	return result;
}

template<class ST> void StringFeatures<ST>::subset_changed_post()
{
	/* max string length has to be updated */
}

template<class ST> ST* StringFeatures<ST>::compute_feature_vector(int32_t num, int32_t& len)
{
	ASSERT(num<get_num_vectors())

	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	len=features[real_num].vlen;
	if (len<=0)
		return NULL;

	ST* target=SG_MALLOC(ST, len);
	sg_memcpy(target, features[real_num].vector, len*sizeof(ST));
	return target;
}

template<class ST> void StringFeatures<ST>::init()
{
	set_generic<ST>();

	alphabet=NULL;
	features.clear();
	single_string=SGVector<ST>();
	order=0;
	preprocess_on_get=false;
	feature_cache=NULL;
	symbol_mask_table=SGVector<ST>();
	num_symbols=0.0;
	original_num_symbols=0;

	SG_ADD(&alphabet, "alphabet", "Alphabet used.");

	/*m_parameters->add(&single_string,
			"single_string",
			"Created by sliding window.");*/
	watch_param("single_string", &single_string);

	SG_ADD(
		&num_symbols, "num_symbols", "Number of used symbols.");
	SG_ADD(
		&original_num_symbols, "original_num_symbols",
		"Original number of used symbols.");
	SG_ADD(
		&order, "order", "Order used in higher order mapping.");
	SG_ADD(
		&preprocess_on_get, "preprocess_on_get", "Preprocess on-the-fly?");
	SG_ADD(
		&symbol_mask_table, "mask_table",
		"Symbol mask table - using in higher order mapping");

	watch_param("string_list", &features);
	watch_method("num_vectors", &StringFeatures::get_num_vectors);
	watch_method("max_string_length", &StringFeatures::get_max_vector_length);
}

/** get feature type the char feature can deal with
 *
 * @return feature type char
 */
template<> EFeatureType StringFeatures<bool>::get_feature_type() const
{
	return F_BOOL;
}

/** get feature type the char feature can deal with
 *
 * @return feature type char
 */
template<> EFeatureType StringFeatures<char>::get_feature_type() const
{
	return F_CHAR;
}

/** get feature type the BYTE feature can deal with
 *
 * @return feature type BYTE
 */
template<> EFeatureType StringFeatures<uint8_t>::get_feature_type() const
{
	return F_BYTE;
}

/** get feature type the SHORT feature can deal with
 *
 * @return feature type SHORT
 */
template<> EFeatureType StringFeatures<int16_t>::get_feature_type() const
{
	return F_SHORT;
}

/** get feature type the WORD feature can deal with
 *
 * @return feature type WORD
 */
template<> EFeatureType StringFeatures<uint16_t>::get_feature_type() const
{
	return F_WORD;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> EFeatureType StringFeatures<int32_t>::get_feature_type() const
{
	return F_INT;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> EFeatureType StringFeatures<uint32_t>::get_feature_type() const
{
	return F_UINT;
}

/** get feature type the LONG feature can deal with
 *
 * @return feature type LONG
 */
template<> EFeatureType StringFeatures<int64_t>::get_feature_type() const
{
	return F_LONG;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> EFeatureType StringFeatures<uint64_t>::get_feature_type() const
{
	return F_ULONG;
}

/** get feature type the SHORTREAL feature can deal with
 *
 * @return feature type SHORTREAL
 */
template<> EFeatureType StringFeatures<float32_t>::get_feature_type() const
{
	return F_SHORTREAL;
}

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> EFeatureType StringFeatures<float64_t>::get_feature_type() const
{
	return F_DREAL;
}

/** get feature type the LONGREAL feature can deal with
 *
 * @return feature type LONGREAL
 */
template<> EFeatureType StringFeatures<floatmax_t>::get_feature_type() const
{
	return F_LONGREAL;
}

template<> bool StringFeatures<bool>::get_masked_symbols(bool symbol, uint8_t mask)
{
	return symbol;
}
template<> float32_t StringFeatures<float32_t>::get_masked_symbols(float32_t symbol, uint8_t mask)
{
	return symbol;
}
template<> float64_t StringFeatures<float64_t>::get_masked_symbols(float64_t symbol, uint8_t mask)
{
	return symbol;
}
template<> floatmax_t StringFeatures<floatmax_t>::get_masked_symbols(floatmax_t symbol, uint8_t mask)
{
	return symbol;
}

template<> bool StringFeatures<bool>::shift_offset(bool symbol, int32_t amount)
{
	return false;
}
template<> float32_t StringFeatures<float32_t>::shift_offset(float32_t symbol, int32_t amount)
{
	return 0;
}
template<> float64_t StringFeatures<float64_t>::shift_offset(float64_t symbol, int32_t amount)
{
	return 0;
}
template<> floatmax_t StringFeatures<floatmax_t>::shift_offset(floatmax_t symbol, int32_t amount)
{
	return 0;
}

template<> bool StringFeatures<bool>::shift_symbol(bool symbol, int32_t amount)
{
	return symbol;
}
template<> float32_t StringFeatures<float32_t>::shift_symbol(float32_t symbol, int32_t amount)
{
	return symbol;
}
template<> float64_t StringFeatures<float64_t>::shift_symbol(float64_t symbol, int32_t amount)
{
	return symbol;
}
template<> floatmax_t StringFeatures<floatmax_t>::shift_symbol(floatmax_t symbol, int32_t amount)
{
	return symbol;
}

#ifndef SUNOS
template<>	template <class CT> bool StringFeatures<float32_t>::obtain_from_char_features(std::shared_ptr<StringFeatures<CT>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
template<>	template <class CT> bool StringFeatures<float64_t>::obtain_from_char_features(std::shared_ptr<StringFeatures<CT>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
template<>	template <class CT> bool StringFeatures<floatmax_t>::obtain_from_char_features(std::shared_ptr<StringFeatures<CT>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
#endif

template<>	void StringFeatures<float32_t>::embed_features(int32_t p_order)
{
}
template<>	void StringFeatures<float64_t>::embed_features(int32_t p_order)
{
}
template<>	void StringFeatures<floatmax_t>::embed_features(int32_t p_order)
{
}

template<>	void StringFeatures<float32_t>::compute_symbol_mask_table(int64_t max_val)
{
}
template<>	void StringFeatures<float64_t>::compute_symbol_mask_table(int64_t max_val)
{
}
template<>	void StringFeatures<floatmax_t>::compute_symbol_mask_table(int64_t max_val)
{
}

template<>	float32_t StringFeatures<float32_t>::embed_word(float32_t* seq, int32_t len)
{
	return 0;
}
template<>	float64_t StringFeatures<float64_t>::embed_word(float64_t* seq, int32_t len)
{
	return 0;
}
template<>	floatmax_t StringFeatures<floatmax_t>::embed_word(floatmax_t* seq, int32_t len)
{
	return 0;
}

template<>	void StringFeatures<float32_t>::unembed_word(float32_t word, uint8_t* seq, int32_t len)
{
}
template<>	void StringFeatures<float64_t>::unembed_word(float64_t word, uint8_t* seq, int32_t len)
{
}
template<>	void StringFeatures<floatmax_t>::unembed_word(floatmax_t word, uint8_t* seq, int32_t len)
{
}
#define LOAD(f_load, sg_type)												\
template<> void StringFeatures<sg_type>::load(std::shared_ptr<File> loader)		\
{																			\
	io::info("loading...");												\
																			\
	SG_SET_LOCALE_C;													\
	SGVector<sg_type>* strs;												\
	int32_t num_str;														\
	int32_t max_len;														\
	loader->f_load(strs, num_str, max_len);									\
	set_features(strs, num_str);											\
	SG_FREE(strs);															\
	SG_RESET_LOCALE;													\
}

LOAD(get_string_list, bool)
LOAD(get_string_list, char)
LOAD(get_string_list, int8_t)
LOAD(get_string_list, uint8_t)
LOAD(get_string_list, int16_t)
LOAD(get_string_list, uint16_t)
LOAD(get_string_list, int32_t)
LOAD(get_string_list, uint32_t)
LOAD(get_string_list, int64_t)
LOAD(get_string_list, uint64_t)
LOAD(get_string_list, float32_t)
LOAD(get_string_list, float64_t)
LOAD(get_string_list, floatmax_t)
#undef LOAD

#define SAVE(f_write, sg_type)												\
template<> void StringFeatures<sg_type>::save(std::shared_ptr<File> writer)		\
{																			\
	if (m_subset_stack->has_subsets())															\
		error("save() is not possible on subset");						\
	SG_SET_LOCALE_C;													\
	ASSERT(writer)															\
	writer->f_write(features.data(), get_num_vectors());				\
	SG_RESET_LOCALE;													\
}

SAVE(set_string_list, bool)
SAVE(set_string_list, char)
SAVE(set_string_list, int8_t)
SAVE(set_string_list, uint8_t)
SAVE(set_string_list, int16_t)
SAVE(set_string_list, uint16_t)
SAVE(set_string_list, int32_t)
SAVE(set_string_list, uint32_t)
SAVE(set_string_list, int64_t)
SAVE(set_string_list, uint64_t)
SAVE(set_string_list, float32_t)
SAVE(set_string_list, float64_t)
SAVE(set_string_list, floatmax_t)
#undef SAVE

template <class ST> template <class CT>
bool StringFeatures<ST>::obtain_from_char_features(std::shared_ptr<StringFeatures<CT>> sf, int32_t start,
		int32_t p_order, int32_t gap, bool rev)
{
	remove_all_subsets();
	ASSERT(sf)

	auto alpha=sf->get_alphabet();
	ASSERT(alpha->get_num_symbols_in_histogram() > 0)

	this->order=p_order;
	cleanup();

	int32_t num_vectors=sf->get_num_vectors();
	ASSERT(num_vectors>0)
	features.reserve(num_vectors);

	SG_DEBUG("{:1.0f} symbols in StringFeatures<*> {} symbols in histogram", sf->get_num_symbols(),
			alpha->get_num_symbols_in_histogram());

	for (int32_t i=0; i<num_vectors; i++)
	{
		int32_t len=-1;
		bool vfree;
		CT* c=sf->get_feature_vector(i, len, vfree);
		ASSERT(!vfree) // won't work when preprocessors are attached

		features.emplace_back(len);

		for (int32_t j=0; j<len; j++)
			features.back()[j]=(ST) alpha->remap_to_bin(c[j]);
	}

	original_num_symbols=alpha->get_num_symbols();
	int32_t max_val=alpha->get_num_bits();

	if (p_order>1)
		num_symbols=Math::powl((floatmax_t) 2, (floatmax_t) max_val*p_order);
	else
		num_symbols=original_num_symbols;
	io::info("max_val (bit): {} order: {} -> results in num_symbols: {:.0f}", max_val, p_order, num_symbols);

	if ( ((floatmax_t) num_symbols) > Math::powl(((floatmax_t) 2),((floatmax_t) sizeof(ST)*8)) )
	{
		error("symbol does not fit into datatype \"{}\" ({})", (char) max_val, (int) max_val);
		return false;
	}

	SG_DEBUG("translate: start={} order={} gap={}(size:{})", start, p_order, gap, sizeof(ST))
	for (int32_t line=0; line<num_vectors; line++)
	{
		int32_t len=0;
		bool vfree;
		ST* fv=get_feature_vector(line, len, vfree);
		ASSERT(!vfree) // won't work when preprocessors are attached

		if (rev)
			Alphabet::translate_from_single_order_reversed(fv, len, start+gap, p_order+gap, max_val, gap);
		else
			Alphabet::translate_from_single_order(fv, len, start+gap, p_order+gap, max_val, gap);

		/* fix the length of the string -- hacky */
		features[line].vlen-=start+gap ;
		if (features[line].vlen<0)
			features[line].vlen=0 ;
	}

	compute_symbol_mask_table(max_val);

	return true;
}

template class StringFeatures<bool>;
template class StringFeatures<char>;
template class StringFeatures<int8_t>;
template class StringFeatures<uint8_t>;
template class StringFeatures<int16_t>;
template class StringFeatures<uint16_t>;
template class StringFeatures<int32_t>;
template class StringFeatures<uint32_t>;
template class StringFeatures<int64_t>;
template class StringFeatures<uint64_t>;
template class StringFeatures<float32_t>;
template class StringFeatures<float64_t>;
template class StringFeatures<floatmax_t>;

template bool StringFeatures<uint16_t>::obtain_from_char_features<uint8_t>(std::shared_ptr<StringFeatures<uint8_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);
template bool StringFeatures<uint32_t>::obtain_from_char_features<uint8_t>(std::shared_ptr<StringFeatures<uint8_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);
template bool StringFeatures<uint64_t>::obtain_from_char_features<uint8_t>(std::shared_ptr<StringFeatures<uint8_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);

template bool StringFeatures<uint16_t>::obtain_from_char_features<uint16_t>(std::shared_ptr<StringFeatures<uint16_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);
template bool StringFeatures<uint32_t>::obtain_from_char_features<uint16_t>(std::shared_ptr<StringFeatures<uint16_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);
template bool StringFeatures<uint64_t>::obtain_from_char_features<uint16_t>(std::shared_ptr<StringFeatures<uint16_t>> sf, int32_t start, int32_t p_order, int32_t gap, bool rev);
}
