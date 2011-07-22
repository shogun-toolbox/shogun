/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Subset support written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Cache.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/io/File.h>
#include <shogun/io/MemoryMappedFile.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Compressor.h>
#include <shogun/base/Parameter.h>

#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/features/Alphabet.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace shogun
{
class CCompressor;
enum E_COMPRESSION_TYPE;
class CAlphabet;
enum EAlphabet;
template <class T> class CDynamicArray;
class CFile;
template <class T> class CMemoryMappedFile;
class CMath;
template <class ST> class CStringPreprocessor;
template <class T> class SGString;

struct SSKDoubleFeature
{
	int feature1;
	int feature2;
	int group;
};

struct SSKTripleFeature
{
	int feature1;
	int feature2;
	int feature3;
	int group;
};

/** @brief Template class StringFeatures implements a list of strings.
 *
 * As this class is a template the underlying storage type is quite arbitrary and
 * not limited to character strings, but could also be sequences of floating
 * point numbers etc. Strings differ from matrices (cf. CSimpleFeatures) in a
 * way that the dimensionality of the feature vectors (i.e. the strings) is not
 * fixed; it may vary between strings.
 *
 * Most string kernels require StringFeatures but a number of them actually
 * requires strings to have same length.
 *
 * When preprocessors are attached to string features they may shorten the
 * string, but are not allowed to return strings longer than max_string_length,
 * as some algorithms depend on this.
 *
 * Also note that string features cannot currently be computed on-the-fly.
 *
 * (Partly) subset access is supported for this feature type.
 * Simple use the (inherited) set_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method
 */
template <class ST> class CStringFeatures : public CFeatures
{
	public:
		/** default constructor
		 *
		 */
        CStringFeatures() : CFeatures(0)
        {
			init();
			alphabet=new CAlphabet();
        }

		/** constructor
		 *
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(EAlphabet alpha) : CFeatures(0)
		{
			init();

			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
		}

		/** constructor
		 *
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(SGStringList<ST> string_list, EAlphabet alpha)
		: CFeatures(0)
		{
			init();

			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
			set_features(string_list.strings, string_list.num_strings, string_list.max_string_length);
		}

		/** constructor
		 *
		 * @param alpha an actual alphabet
		 */
		CStringFeatures(SGStringList<ST> string_list, CAlphabet* alpha)
		: CFeatures(0)
		{
			init();

			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
			set_features(string_list.strings, string_list.num_strings, string_list.max_string_length);
		}

		/** constructor
		 *
		 * @param alpha alphabet to use for string features
		 */
		CStringFeatures(CAlphabet* alpha)
		: CFeatures(0)
		{
			init();

			ASSERT(alpha);
			SG_REF(alpha);
			alphabet=alpha;
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
		}

		/** copy constructor */
		CStringFeatures(const CStringFeatures & orig)
		: CFeatures(orig), num_vectors(orig.num_vectors),
			single_string(orig.single_string),
			length_of_single_string(orig.length_of_single_string),
			max_string_length(orig.max_string_length),
			num_symbols(orig.num_symbols),
			original_num_symbols(orig.original_num_symbols),
			order(orig.order), preprocess_on_get(false),
			feature_cache(NULL)
		{
			init();

			ASSERT(orig.single_string == NULL); //not implemented

			alphabet=orig.alphabet;
			SG_REF(alphabet);

			if (orig.features)
			{
				features=new SGString<ST>[orig.num_vectors];

				for (int32_t i=0; i<num_vectors; i++)
				{
					features[i].string=new ST[orig.features[i].length];
					features[i].length=orig.features[i].length;
					memcpy(features[i].string, orig.features[i].string, sizeof(ST)*orig.features[i].length);
				}
			}

			if (orig.symbol_mask_table)
			{
				symbol_mask_table=new ST[256];
				for (int32_t i=0; i<256; i++)
					symbol_mask_table[i]=orig.symbol_mask_table[i];
			}

			m_subset=orig.m_subset->duplicate();
		}

		/** constructor
		 *
		 * @param loader File object via which to load data
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(CFile* loader, EAlphabet alpha=DNA)
		: CFeatures(loader), num_vectors(0),
		  features(NULL), single_string(NULL), length_of_single_string(0),
		  max_string_length(0), order(0),
		  symbol_mask_table(NULL), preprocess_on_get(false), feature_cache(NULL)
		{
			init();

			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
			load(loader);
		}

		virtual ~CStringFeatures()
		{
			cleanup();

			SG_UNREF(alphabet);
		}

		/** cleanup string features.
		 *
		 * removes any subset before
		 *
		 * */
		virtual void cleanup()
		{
			remove_subset();

			if (single_string)
			{
				delete[] single_string;
				single_string=NULL;
			}
			else
			{
				for (int32_t i=0; i<num_vectors; i++)
					cleanup_feature_vector(i);
			}

			num_vectors=0;
			delete[] features;
			delete[] symbol_mask_table;
			features=NULL;
			symbol_mask_table=NULL;

			/* start with a fresh alphabet, but instead of emptying the histogram
			 * create a new object (to leave the alphabet object alone if it is used
			 * by others)
			 */
			CAlphabet* alpha=new CAlphabet(alphabet->get_alphabet());
			SG_UNREF(alphabet);
			alphabet=alpha;
			SG_REF(alphabet);
		}

		/** cleanup a single feature vector
		 *
		 * possible with subset
		 *
		 * @param num number of the vector
		 * */
		virtual void cleanup_feature_vector(int32_t num)
		{
			ASSERT(num<get_num_vectors());

			if (features)
			{
				int32_t real_num=subset_idx_conversion(num);
				delete[] features[real_num].string;
				features[real_num].string=NULL;
				features[real_num].length=0;

				determine_maximum_string_length();
			}
		}

		/** get feature class
		 *
		 * @return feature class STRING
		 */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }

		/** get feature type
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type() { return F_UNKNOWN; }

		/** get alphabet used in string features
		 *
		 * @return alphabet
		 */
		inline CAlphabet* get_alphabet()
		{
			SG_REF(alphabet);
			return alphabet;
		}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const
		{
			return new CStringFeatures<ST>(*this);
		}

		/** get string for selected example num
		 *
		 * possible with subset
		 *
		 * @param num index of the string
		 */
		SGVector<ST> get_feature_vector(int32_t num)
		{
			ASSERT(features);
			if (num>=get_num_vectors())
			{
				SG_ERROR("Index out of bounds (number of strings %d, you "
						"requested %d)\n", get_num_vectors(), num);
			}

			int32_t l;
			bool free_vec;
			ST* vec=get_feature_vector(num, l, free_vec);
			ST* dst=(ST*) SG_MALLOC(l*sizeof(ST));
			memcpy(dst, vec, l*sizeof(ST));
			free_feature_vector(vec, num, free_vec);
			return SGVector<ST>(dst,l);
		}

		/** set string for selected example num
		 *
		 * not possible with subset
		 *
		 * @param src destination where vector will be copied from
		 * @param len number of features in vector
		 * @param num index of the string
		 */
		void set_feature_vector(SGVector<ST> vector, int32_t num)
		{
			ASSERT(features);

			if (m_subset)
				SG_ERROR("A subset is set, cannot set feature vector\n");

			if (num>=num_vectors)
			{
				SG_ERROR("Index out of bounds (number of strings %d, you "
						"requested %d)\n", num_vectors, num);
			}

			if (vector.vlen<=0)
				SG_ERROR("String has zero or negative length\n");

			cleanup_feature_vector(num);
			features[num].length=vector.vlen;
			features[num].string=new ST[vector.vlen];
			memcpy(features[num].string, vector.vector, vector.vlen*sizeof(ST));

			determine_maximum_string_length();
		}

		/** call this to preprocess string features upon get_feature_vector
		 */
		void enable_on_the_fly_preprocessing()
		{
			preprocess_on_get=true;
		}

		/** call this to disable on the fly feature preprocessing on
		 * get_feature_vector. Useful when you manually apply preprocessors.
		 */
		void disable_on_the_fly_preprocessing()
		{
			preprocess_on_get=false;
		}

		/** get feature vector for sample num
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @param len length is returned by reference
		 * @param dofree whether returned vector must be freed by
		 * caller via free_feature_vector
		 * @return feature vector for sample num
		 */
		ST* get_feature_vector(int32_t num, int32_t& len, bool& dofree)
		{
			ASSERT(features);
			ASSERT(num<get_num_vectors());


			int32_t real_num=subset_idx_conversion(num);

			if (!preprocess_on_get)
			{
				dofree=false;
				len=features[real_num].length;
				return features[real_num].string;
			}
			else
			{
				SG_DEBUG( "computing feature vector!\n") ;
				ST* feat=compute_feature_vector(num, len);
				dofree=true;

				if (get_num_preprocessors())
				{
					ST* tmp_feat_before=feat;

					for (int32_t i=0; i<get_num_preprocessors(); i++)
					{
						CStringPreprocessor<ST>* p=(CStringPreprocessor<ST>*) get_preprocessor(i);
						feat=p->apply_to_string(tmp_feat_before, len);
						SG_UNREF(p);
						delete[] tmp_feat_before;
						tmp_feat_before=feat;
					}
				}
				// TODO: implement caching
				return feat;
			}
		}

		/** get a transposed copy of the features
		 *
		 * possible with subset
		 *
		 * @return transposed copy
		 */
		CStringFeatures<ST>* get_transposed()
		{
			int32_t num_feat;
			int32_t num_vec;
			SGString<ST>* s=get_transposed(num_feat, num_vec);
			SGStringList<ST> string_list;
			string_list.strings = s;
			string_list.num_strings = num_vec;
			string_list.max_string_length = num_feat;

			return new CStringFeatures<ST>(string_list, alphabet);
		}

		/** compute and return the transpose of string features matrix
		 * which will be prepocessed.
		 * num_feat, num_vectors are returned by reference
		 * caller has to clean up
		 *
		 * note that strings all have to have same length
		 *
		 * possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return transposed string features
		 */
		SGString<ST>* get_transposed(int32_t &num_feat, int32_t &num_vec)
		{
			num_feat=get_num_vectors();
			num_vec=get_max_vector_length();
			ASSERT(have_same_length());

			SG_DEBUG("Allocating memory for transposed string features of size %ld\n",
					int64_t(num_feat)*num_vec);

			SGString<ST>* sf=new SGString<ST>[num_vec];

			for (int32_t i=0; i<num_vec; i++)
			{
				sf[i].string=new ST[num_feat];
				sf[i].length=num_feat;
			}

			for (int32_t i=0; i<num_feat; i++)
			{
				int32_t len=0;
				bool free_vec=false;
				ST* vec=get_feature_vector(i, len, free_vec);

				for (int32_t j=0; j<num_vec; j++)
					sf[j].string[i]=vec[j];

				free_feature_vector(vec, i, free_vec);
			}
			return sf;
		}

		/** free feature vector
		 *
		 * possible with subset
		 *
		 * @param feat_vec feature vector to free
		 * @param num index in feature cache, possibly from subset
		 * @param dofree if vector should be really deleted
		 */
		void free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
		{
			if (num>=get_num_vectors())
			{
				SG_ERROR(
					"Trying to access string[%d] but num_str=%d\n", num,
					get_num_vectors());
			}

			int32_t real_num=subset_idx_conversion(num);

			if (feature_cache)
				feature_cache->unlock_entry(real_num);

			if (dofree)
				delete[] feat_vec ;
		}

		/** get feature
		 *
		 * possible with subset
		 *
		 * @param vec_num which vector
		 * @param feat_num which feature, possibly from subset
		 * @return feature
		 */
		virtual ST inline get_feature(int32_t vec_num, int32_t feat_num)
		{
			ASSERT(vec_num<get_num_vectors());

			int32_t len;
			bool free_vec;
			ST* vec=get_feature_vector(vec_num, len, free_vec);
			ASSERT(feat_num<len);
			ST result=vec[feat_num];
			free_feature_vector(vec, vec_num, free_vec);

			return result;
		}

		/** get vector length
		 *
		 * possible with subset
		 *
		 * @param vec_num which vector, possibly from subset
		 * @return length of vector
		 */
		virtual inline int32_t get_vector_length(int32_t vec_num)
		{
			ASSERT(vec_num<get_num_vectors());

			int32_t len;
			bool free_vec;
			ST* vec=get_feature_vector(vec_num, len, free_vec);
			free_feature_vector(vec, vec_num, free_vec);
			return len;
		}

		/** get maximum vector length
		 *
		 * this one is updated when a subset is set
		 *
		 * @return maximum vector/string length
		 */
		virtual inline int32_t get_max_vector_length()
		{
			return max_string_length;
		}

		/** @return number of vectors, possibly of subset */
		virtual inline int32_t get_num_vectors() const
		{
			return m_subset ? m_subset->get_size() : num_vectors;
		}

		/** get number of symbols
		 *
		 * Note: floatmax_t sounds weird, but LONG is not long enough
		 *
		 * @return number of symbols
		 */
		inline floatmax_t get_num_symbols() { return num_symbols; }

		/** get maximum number of symbols
		 *
		 * Note: floatmax_t sounds weird, but int64_t is not long enough (and
		 * there is no int128_t type)
		 *
		 * @return maximum number of symbols
		 */
		inline floatmax_t get_max_num_symbols() { return CMath::powl(2,sizeof(ST)*8); }

		// these functions are necessary to find out about a former conversion process

		/** number of symbols before higher order mapping
		 *
		 * @return original number of symbols
		 */
		inline floatmax_t get_original_num_symbols() { return original_num_symbols; }

		/** order used for higher order mapping
		 *
		 * @return order
		 */
		inline int32_t get_order() { return order; }

		/** a higher order mapped symbol will be shaped such that the symbols
		 * specified by bits in the mask will be returned.
		 *
		 * @param symbol symbol to mask
		 * @param mask mask to apply
		 * @return masked symbol
		 */
		inline ST get_masked_symbols(ST symbol, uint8_t mask)
		{
			ASSERT(symbol_mask_table);
			return symbol_mask_table[mask] & symbol;
		}

		/** shift offset to the left by amount
		 *
		 * @param offset offset to shift
		 * @param amount amount to shift the offset
		 * @return shifted offset
		 */
		inline ST shift_offset(ST offset, int32_t amount)
		{
			ASSERT(alphabet);
			return (offset << (amount*alphabet->get_num_bits()));
		}

		/** shift symbol to the right by amount (taking care of custom symbol sizes)
		 *
		 * @param symbol symbol to shift
		 * @param amount amount to shift the symbol
		 * @return shifted symbol
		 */
		inline ST shift_symbol(ST symbol, int32_t amount)
		{
			ASSERT(alphabet);
			return (symbol >> (amount*alphabet->get_num_bits()));
		}

		/** load features from file
		 *
		 * @param loader File object via which to load data
		 */
		virtual inline void load(CFile* loader);

		/** load ascii line-based string features from file.
		 *
		 * any subset is removed before
		 *
		 * @param fname filename to load from
		 * @param remap_to_bin if translation to other binary alphabet
		 * should be performed
		 * @param ascii_alphabet src alphabet
		 * @param binary_alphabet alphabet to translate to
		 */
		void load_ascii_file(char* fname, bool remap_to_bin=true,
				EAlphabet ascii_alphabet=DNA, EAlphabet binary_alphabet=RAWDNA)
		{
			remove_subset();

			size_t blocksize=1024*1024;
			size_t required_blocksize=0;
			uint8_t* dummy=new uint8_t[blocksize];
			uint8_t* overflow=NULL;
			int32_t overflow_len=0;

			cleanup();

			CAlphabet* alpha=new CAlphabet(ascii_alphabet);
			CAlphabet* alpha_bin=new CAlphabet(binary_alphabet);

			FILE* f=fopen(fname, "ro");

			if (f)
			{
				num_vectors=0;
				max_string_length=0;

				SG_INFO("counting line numbers in file %s\n", fname);
				size_t block_offs=0;
				size_t old_block_offs=0;
				fseek(f, 0, SEEK_END);
				size_t fsize=ftell(f);
				rewind(f);

				if (blocksize>fsize)
					blocksize=fsize;

				SG_DEBUG("block_size=%ld file_size=%ld\n", blocksize, fsize);

				size_t sz=blocksize;
				while (sz == blocksize)
				{
					sz=fread(dummy, sizeof(uint8_t), blocksize, f);
					bool contains_cr=false;
					for (size_t i=0; i<sz; i++)
					{
						block_offs++;
						if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
						{
							num_vectors++;
							contains_cr=true;
							required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
							old_block_offs=block_offs;
						}
					}
					SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
				}

				SG_INFO("found %d strings\n", num_vectors);
				delete[] dummy;
				blocksize=required_blocksize;
				dummy=new uint8_t[blocksize];
				overflow=new uint8_t[blocksize];
				features=new SGString<ST>[num_vectors];

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
							//SG_PRINT("i:%d len:%d old_sz:%d\n", i, len, old_sz);
							max_string_length=CMath::max(max_string_length, len+overflow_len);

							features[lines].length=len;
							features[lines].string=new ST[len];

							if (remap_to_bin)
							{
								for (int32_t j=0; j<overflow_len; j++)
									features[lines].string[j]=alpha->remap_to_bin(overflow[j]);
								for (int32_t j=0; j<len; j++)
									features[lines].string[j+overflow_len]=alpha->remap_to_bin(dummy[old_sz+j]);
								alpha->add_string_to_histogram(&dummy[old_sz], len);
								alpha_bin->add_string_to_histogram(features[lines].string, features[lines].length);
							}
							else
							{
								for (int32_t j=0; j<overflow_len; j++)
									features[lines].string[j]=overflow[j];
								for (int32_t j=0; j<len; j++)
									features[lines].string[j+overflow_len]=dummy[old_sz+j];
								alpha->add_string_to_histogram(&dummy[old_sz], len);
								alpha->add_string_to_histogram(features[lines].string, features[lines].length);
							}

							// clear overflow
							overflow_len=0;

							//CMath::display_vector(features[lines].string, len);
							old_sz=i+1;
							lines++;
							SG_PROGRESS(lines, 0, num_vectors, 1, "LOADING:\t");
						}
					}
					for (size_t i=old_sz; i<sz; i++)
						overflow[i-old_sz]=dummy[i];

					overflow_len=sz-old_sz;
				}

				if (alpha->check_alphabet_size() && alpha->check_alphabet())
				{
					SG_INFO("file successfully read\n");
					SG_INFO("max_string_length=%d\n", max_string_length);
					SG_INFO("num_strings=%d\n", num_vectors);
				}
				fclose(f);
			}

			delete[] dummy;

			SG_UNREF(alphabet);

			if (remap_to_bin)
				alphabet=alpha_bin;
			else
				alphabet=alpha;
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
		}

		/** load fasta file as string features
		 *
		 * any subset is removed before
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @return if loading was successful
		 */
		bool load_fasta_file(const char* fname, bool ignore_invalid=false)
		{
			remove_subset();

			int32_t i=0;
			uint64_t len=0;
			uint64_t offs=0;
			int32_t num=0;
			int32_t max_len=0;

			CMemoryMappedFile<char> f(fname);

			while (true)
			{
				char* s=f.get_line(len, offs);
				if (!s)
					break;

				if (len>0 && s[0]=='>')
					num++;
			}

			if (num==0)
				SG_ERROR("No fasta hunks (lines starting with '>') found\n");

			cleanup();
			SG_UNREF(alphabet);
			alphabet=new CAlphabet(DNA);
			num_symbols=alphabet->get_num_symbols();

			SGString<ST>* strings=new SGString<ST>[num];
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
						SG_ERROR("Error reading fasta entry in line %d len=%ld", 4*i+1, len);

					if (s[0]=='>' || offs==f.get_size())
					{
						offs-=len+1; // seek to beginning
						if (offs==f.get_size())
						{
							SG_DEBUG("at EOF\n");
							fasta_len+=len;
						}

						len=fasta_len-spanned_lines;
						strings[i].string=new ST[len];
						strings[i].length=len;

						ST* str=strings[i].string;
						int32_t idx=0;
						SG_DEBUG("'%.*s', len=%d, spanned_lines=%d\n", (int32_t) id_len, id, (int32_t) len, (int32_t) spanned_lines);

						for (int32_t j=0; j<fasta_len; j++)
						{
							if (fasta[j]=='\n')
								continue;

							ST c=(ST) fasta[j];

							if (ignore_invalid  && !alphabet->is_valid((uint8_t) fasta[j]))
								c=(ST) 'A';

							if (idx>=len)
								SG_ERROR("idx=%d j=%d fasta_len=%d, spanned_lines=%d str='%.*s'\n", idx, j, fasta_len, spanned_lines, idx, str);
							str[idx++]=c;
						}
						max_len=CMath::max(max_len, strings[i].length);


						break;
					}

					spanned_lines++;
					fasta_len+=len+1; // including '\n'
					s=f.get_line(len, offs);
				}
			}
			return set_features(strings, num, max_len);
		}

		/** load fastq file as string features
		 *
		 * removes subset beforehand
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @param bitremap_in_single_string if set to true, do binary embedding of symbols
		 * @return if loading was successful
		 */
		bool load_fastq_file(const char* fname,
				bool ignore_invalid=false, bool bitremap_in_single_string=false)
		{
			remove_subset();

			CMemoryMappedFile<char> f(fname);

			int32_t i=0;
			uint64_t len=0;
			uint64_t offs=0;

			int32_t num=f.get_num_lines();
			int32_t max_len=0;

			if (num%4)
				SG_ERROR("Number of lines must be divisible by 4 in fastq files\n");
			num/=4;

			cleanup();
			SG_UNREF(alphabet);
			alphabet=new CAlphabet(DNA);

			SGString<ST>* strings;

			ST* str;
			if (bitremap_in_single_string)
			{
				strings=new SGString<ST>[1];
				strings[0].string=new ST[num];
				strings[0].length=num;
				f.get_line(len, offs);
				f.get_line(len, offs);
				order=len;
				max_len=num;
				offs=0;
				original_num_symbols=alphabet->get_num_symbols();
				int32_t max_val=alphabet->get_num_bits();
				str=new ST[len];
			}
			else
				strings=new SGString<ST>[num];

			for (i=0;i<num; i++)
			{
				if (!f.get_line(len, offs))
					SG_ERROR("Error reading 'read' identifier in line %d", 4*i);

				char* s=f.get_line(len, offs);
				if (!s || len==0)
					SG_ERROR("Error reading 'read' in line %d len=%ld", 4*i+1, len);

				if (bitremap_in_single_string)
				{
					if (len!=order)
						SG_ERROR("read in line %d not of length %d (is %d)\n", 4*i+1, order, len);
					for (int32_t j=0; j<order; j++)
						str[j]=(ST) alphabet->remap_to_bin((uint8_t) s[j]);

					strings[0].string[i]=embed_word(str, order);
				}
				else
				{
					strings[i].string=new ST[len];
					strings[i].length=len;
					str=strings[i].string;

					if (ignore_invalid)
					{
						for (int32_t j=0; j<len; j++)
						{
							if (alphabet->is_valid((uint8_t) s[j]))
								str[j]= (ST) s[j];
							else
								str[j]= (ST) 'A';
						}
					}
					else
					{
						for (int32_t j=0; j<len; j++)
							str[j]= (ST) s[j];
					}
					max_len=CMath::max(max_len, (int32_t) len);
				}


				if (!f.get_line(len, offs))
					SG_ERROR("Error reading 'read' quality identifier in line %d", 4*i+2);

				if (!f.get_line(len, offs))
					SG_ERROR("Error reading 'read' quality in line %d", 4*i+3);
			}

			if (bitremap_in_single_string)
				num=1;

			num_vectors=num;
			max_string_length=max_len;
			features=strings;

			return true;
		}

		/** load features from directory
		 *
		 *removes subset before
		 *
		 * @param dirname directory name to load from
		 * @return if loading was successful
		 */
		bool load_from_directory(char* dirname)
		{
			remove_subset();

			struct dirent **namelist;
			int32_t n;

            SGIO::set_dirname(dirname);

			SG_DEBUG("dirname '%s'\n", dirname);

			n=scandir(dirname, &namelist, &SGIO::filter, alphasort);
			if (n <= 0)
			{
				SG_ERROR("error calling scandir - no files found\n");
				return false;
			}
			else
			{
				SGString<ST>* strings=NULL;

				int32_t num=0;
				int32_t max_len=-1;

				//usually n==num_vec, but it might not in race conditions
				//(file perms modified, file erased)
				strings=new SGString<ST>[n];

				for (int32_t i=0; i<n; i++)
				{
					char* fname=SGIO::concat_filename(namelist[i]->d_name);

					struct stat s;
					off_t filesize=0;

					if (!stat(fname, &s) && s.st_size>0)
					{
						filesize=s.st_size/sizeof(ST);

						FILE* f=fopen(fname, "ro");
						if (f)
						{
							ST* str=new ST[filesize];
							SG_DEBUG("%s:%ld\n", fname, (int64_t) filesize);
							if (fread(str, sizeof(ST), filesize, f)!=(size_t) filesize)
								SG_ERROR("failed to read file\n");
							strings[num].string=str;
							strings[num].length=filesize;
							max_len=CMath::max(max_len, strings[num].length);

							num++;
							fclose(f);
						}
					}
					else
						SG_ERROR("empty or non readable file \'%s\'\n", fname);

					SG_FREE(namelist[i]);
				}
				SG_FREE(namelist);

				if (num>0 && strings)
				{
					set_features(strings, num, max_len);
					return true;
				}
			}
			return false;
		}

		/** set features
		 *
		 * not possible with subset
		 *
		 */
        void set_features(SGStringList<ST> feats)
        {
            set_features(feats.strings, feats.num_strings, feats.max_string_length);
        }

		/** set features
		 *
		 * not possible with subset
		 *
		 * @param p_features new features
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 * @return if setting was successful
		 */
		bool set_features(SGString<ST>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)
		{
			if (m_subset)
				SG_ERROR("Cannot call set_features() with subset.\n");

			if (p_features)
			{
				CAlphabet* alpha=new CAlphabet(alphabet->get_alphabet());

				//compute histogram for char/byte
				for (int32_t i=0; i<p_num_vectors; i++)
					alpha->add_string_to_histogram( p_features[i].string, p_features[i].length);

				SG_INFO("max_value_in_histogram:%d\n", alpha->get_max_value_in_histogram());
				SG_INFO("num_symbols_in_histogram:%d\n", alpha->get_num_symbols_in_histogram());

				if (alpha->check_alphabet_size() && alpha->check_alphabet())
				{
					cleanup();
					SG_UNREF(alphabet);

					alphabet=alpha;
					SG_REF(alphabet);

					features=p_features;
					num_vectors=p_num_vectors;
					max_string_length=p_max_string_length;

					return true;
				}
				else
					SG_UNREF(alpha);
			}

			return false;
		}

		/** append features
		 * If the given string features have a subset, only this will be copied
		 *
		 * not possible with subset
		 *
		 * @param sf features to append
		 * @return if setting was successful
		 */
		bool append_features(CStringFeatures<ST>* sf)
		{
			ASSERT(sf);

			if (m_subset)
				SG_ERROR("Cannot call set_features() with subset.\n");

			SGString<ST>* new_features=new SGString<ST>[sf->get_num_vectors()];

			index_t sf_num_str=sf->get_num_vectors();
			for (int32_t i=0; i<sf_num_str; i++)
			{
				int32_t real_i = sf->subset_idx_conversion(i);
				int32_t length=sf->features[real_i].length;
				new_features[i].string=new ST[length];
				memcpy(new_features[i].string, sf->features[real_i].string, length);
				new_features[i].length=length;
			}
			return append_features(new_features, sf_num_str,
					sf->max_string_length);
		}

		/** append features
		 *
		 *  not possible with subset
		 *
		 * @param p_features features to append
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 *
		 * note that p_features will be delete[]'d on success
		 *
		 * @return if setting was successful
		 */
		bool append_features(SGString<ST>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)
        {
			if (m_subset)
				SG_ERROR("Cannot call set_features() with subset.\n");

            if (!features)
                return set_features(p_features, p_num_vectors, p_max_string_length);

            CAlphabet* alpha=new CAlphabet(alphabet->get_alphabet());

            //compute histogram for char/byte
            for (int32_t i=0; i<p_num_vectors; i++)
                alpha->add_string_to_histogram( p_features[i].string, p_features[i].length);

            SG_INFO("max_value_in_histogram:%d\n", alpha->get_max_value_in_histogram());
            SG_INFO("num_symbols_in_histogram:%d\n", alpha->get_num_symbols_in_histogram());

            if (alpha->check_alphabet_size() && alpha->check_alphabet())
            {
                SG_UNREF(alpha);
                for (int32_t i=0; i<p_num_vectors; i++)
                    alphabet->add_string_to_histogram( p_features[i].string, p_features[i].length);

                int32_t old_num_vectors=num_vectors;
                num_vectors=old_num_vectors+p_num_vectors;
                SGString<ST>* new_features=new SGString<ST>[num_vectors];

                for (int32_t i=0; i<num_vectors; i++)
                {
                    if (i<old_num_vectors)
                    {
                        new_features[i].string=features[i].string;
                        new_features[i].length=features[i].length;
                    }
                    else
                    {
                        new_features[i].string=p_features[i-old_num_vectors].string;
                        new_features[i].length=p_features[i-old_num_vectors].length;
                    }
                }
                delete[] features;
				delete[] p_features; // free now obsolete features

                this->features=new_features;
                max_string_length=CMath::max(max_string_length, p_max_string_length);

                return true;
            }
            SG_UNREF(alpha);

            return false;
        }

        SGStringList<ST> get_features()
        {
            SGStringList<ST> sl;

            sl.strings=get_features(sl.num_strings, sl.max_string_length);
            return sl;
        }

		/** get_features
		 *
		 * not possible with subset
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual SGString<ST>* get_features(int32_t& num_str, int32_t& max_str_len)
		{
			if (m_subset)
				SG_ERROR("get features() is not possible on subset");

			num_str=num_vectors;
			max_str_len=max_string_length;
			return features;
		}

		/** copy_features
		 *
		 * possible with subset
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual SGString<ST>* copy_features(int32_t& num_str, int32_t& max_str_len)
		{
			ASSERT(num_vectors>0);

			num_str=get_num_vectors();
			max_str_len=max_string_length;
			SGString<ST>* new_feat=new SGString<ST>[num_str];

			for (int32_t i=0; i<num_str; i++)
			{
				int32_t len;
				bool free_vec;
				ST* vec=get_feature_vector(i, len, free_vec);
				new_feat[i].string=new ST[len];
				new_feat[i].length=len;
				memcpy(new_feat[i].string, vec, ((size_t) len) * sizeof(ST));
				free_feature_vector(vec, i, free_vec);
			}

			return new_feat;
		}

		/** get_features  (swig compatible)
		 *
		 * possible with subset
		 *
		 * @param dst string features (returned)
		 * @param num_str number of strings (returned)
		 */
		virtual void get_features(SGString<ST>** dst, int32_t* num_str)
		{
			int32_t num_vec;
			int32_t max_str_len;
			*dst=copy_features(num_vec, max_str_len);
			*num_str=num_vec;
		}

		/** save features to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object via which to save data
		 */
		virtual inline void save(CFile* writer);

		/** load compressed features from file
		 *
		 * any subset is removed before
		 *
		 * @param src filename to load from
		 * @param decompress whether to decompress on loading
		 * @return if loading was successful
		 */
		virtual bool load_compressed(char* src, bool decompress)
		{
			remove_subset();

			FILE* file=NULL;

			if (!(file=fopen(src, "r")))
				return false;
			cleanup();

			// header shogun v0
			char id[4];
			if (fread(&id[0], sizeof(char), 1, file)!=1)
				SG_ERROR("failed to read header");
			ASSERT(id[0]=='S');
			if (fread(&id[1], sizeof(char), 1, file)!=1)
				SG_ERROR("failed to read header");
			ASSERT(id[1]=='G');
			if (fread(&id[2], sizeof(char), 1, file)!=1)
				SG_ERROR("failed to read header");
			ASSERT(id[2]=='V');
			if (fread(&id[3], sizeof(char), 1, file)!=1)
				SG_ERROR("failed to read header");
			ASSERT(id[3]=='0');

			//compression type
			uint8_t c;
			if (fread(&c, sizeof(uint8_t), 1, file)!=1)
				SG_ERROR("failed to read compression type");
			CCompressor* compressor= new CCompressor((E_COMPRESSION_TYPE) c);
			//alphabet
			uint8_t a;
			delete alphabet;
			if (fread(&a, sizeof(uint8_t), 1, file)!=1)
				SG_ERROR("failed to read compression alphabet");
			alphabet=new CAlphabet((EAlphabet) a);
			// number of vectors
			if (fread(&num_vectors, sizeof(int32_t), 1, file)!=1)
				SG_ERROR("failed to read compression number of vectors");
			ASSERT(num_vectors>0);
			// maximum string length
			if (fread(&max_string_length, sizeof(int32_t), 1, file)!=1)
				SG_ERROR("failed to read maximum string length");
			ASSERT(max_string_length>0);

			features=new SGString<ST>[num_vectors];

			// vectors
			for (int32_t i=0; i<num_vectors; i++)
			{
				// vector len compressed
				int32_t len_compressed;
				if (fread(&len_compressed, sizeof(int32_t), 1, file)!=1)
					SG_ERROR("failed to read vector length compressed");
				// vector len uncompressed
				int32_t len_uncompressed;
				if (fread(&len_uncompressed, sizeof(int32_t), 1, file)!=1)
					SG_ERROR("failed to read vector length uncompressed");

				// vector raw data
				if (decompress)
				{
					features[i].string=new ST[len_uncompressed];
					features[i].length=len_uncompressed;
					uint8_t* compressed=new uint8_t[len_compressed];
					if (fread(compressed, sizeof(uint8_t), len_compressed, file)!=(size_t) len_compressed)
						SG_ERROR("failed to read compressed data (expected %d bytes)", len_compressed);
					uint64_t uncompressed_size=len_uncompressed;
					uncompressed_size*=sizeof(ST);
					compressor->decompress(compressed, len_compressed,
							(uint8_t*) features[i].string, uncompressed_size);
					delete[] compressed;
					ASSERT(uncompressed_size==((uint64_t) len_uncompressed)*sizeof(ST));
				}
				else
				{
					int32_t offs=CMath::ceil(2.0*sizeof(int32_t)/sizeof(ST));
					features[i].string=new ST[len_compressed+offs];
					features[i].length=len_compressed+offs;
					int32_t* feat32ptr=((int32_t*) (features[i].string));
					memset(features[i].string, 0, offs*sizeof(ST));
					feat32ptr[0]=(int32_t) len_compressed;
					feat32ptr[1]=(int32_t) len_uncompressed;
					uint8_t* compressed=(uint8_t*) (&features[i].string[offs]);
					if (fread(compressed, 1, len_compressed, file)!=(size_t) len_compressed)
						SG_ERROR("failed to read uncompressed data");
				}
			}

			delete compressor;
			fclose(file);

			return false;
		}

		/** save compressed features to file
		 *
		 * not possible with subset
		 *
		 * @param dest filename to save to
		 * @param compression compressor to use
		 * @param level compression level to use (1-9)
		 * @return if saving was successful
		 */
		virtual bool save_compressed(char* dest, E_COMPRESSION_TYPE compression, int level)
		{
			if (m_subset)
				SG_ERROR("save_compressed() is not possible on subset");

			FILE* file=NULL;

			if (!(file=fopen(dest, "wb")))
				return false;

			CCompressor* compressor= new CCompressor(compression);

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
				delete[] compressed;

				free_feature_vector(vec, i, vfree);
			}

			delete compressor;
			fclose(file);
			return true;
		}


		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size() { return sizeof(ST); }

		/** apply preprocessor
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preprocessor(bool force_preprocessing=false)
		{
			SG_DEBUG( "force: %d\n", force_preprocessing);

			for (int32_t i=0; i<get_num_preprocessors(); i++)
			{
				if ( (!is_preprocessed(i) || force_preprocessing) )
				{
					set_preprocessed(i);
					CStringPreprocessor<ST>* p=(CStringPreprocessor<ST>*) get_preprocessor(i);
					SG_INFO( "preprocessing using preproc %s\n", p->get_name());

					if (!p->apply_to_string_features(this))
					{
						SG_UNREF(p);
						return false;
					}
					else
						SG_UNREF(p);
				}
			}
			return true;
		}

		/** slides a window of size window_size over the current single string
		 * step_size is the amount by which the window is shifted.
		 * creates (string_len-window_size)/step_size many feature obj
		 * if skip is nonzero, skip the first 'skip' characters of each string
		 *
		 * not implemented for subset
		 *
		 * @param window_size window size
		 * @param step_size step size
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_sliding_window(int32_t window_size, int32_t step_size, int32_t skip=0)
		{
			if (m_subset)
				SG_NOTIMPLEMENTED;

			ASSERT(step_size>0);
			ASSERT(window_size>0);
			ASSERT(num_vectors==1 || single_string);
			ASSERT(max_string_length>=window_size ||
					(single_string && length_of_single_string>=window_size));

			//in case we are dealing with a single remapped string
			//allow remapping
			if (single_string)
				num_vectors= (length_of_single_string-window_size)/step_size + 1;
			else if (num_vectors==1)
			{
				num_vectors= (max_string_length-window_size)/step_size + 1;
				length_of_single_string=max_string_length;
			}

			SGString<ST>* f=new SGString<ST>[num_vectors];
			int32_t offs=0;
			for (int32_t i=0; i<num_vectors; i++)
			{
				f[i].string=&features[0].string[offs+skip];
				f[i].length=window_size-skip;
				offs+=step_size;
			}
			single_string=features[0].string;
			delete[] features;
			features=f;
			max_string_length=window_size-skip;

			return num_vectors;
		}

		/** extracts windows of size window_size from first string
		 * using the positions in list
		 *
		 * not implemented for subset
		 *
		 * @param window_size window size
		 * @param positions positions
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_position_list(int32_t window_size, CDynamicArray<int32_t>* positions, int32_t skip=0)
		{
			if (m_subset)
				SG_NOTIMPLEMENTED;

			ASSERT(positions);
			ASSERT(window_size>0);
			ASSERT(num_vectors==1 || single_string);
			ASSERT(max_string_length>=window_size ||
					(single_string && length_of_single_string>=window_size));

			num_vectors= positions->get_num_elements();
			ASSERT(num_vectors>0);

			int32_t len;

			//in case we are dealing with a single remapped string
			//allow remapping
			if (single_string)
				len=length_of_single_string;
			else
			{
				single_string=features[0].string;
				len=max_string_length;
				length_of_single_string=max_string_length;
			}

			SGString<ST>* f=new SGString<ST>[num_vectors];
			for (int32_t i=0; i<num_vectors; i++)
			{
				int32_t p=positions->get_element(i);

				if (p>=0 && p<=len-window_size)
				{
					f[i].string=&features[0].string[p+skip];
					f[i].length=window_size-skip;
				}
				else
				{
					num_vectors=1;
					max_string_length=len;
					features[0].length=len;
					single_string=NULL;
					delete[] f;
					SG_ERROR("window (size:%d) starting at position[%d]=%d does not fit in sequence(len:%d)\n",
							window_size, i, p, len);
					return -1;
				}
			}

			delete[] features;
			features=f;
			max_string_length=window_size-skip;

			return num_vectors;
		}

		/** obtain string features from char features
		 *
		 * wrapper for template method
		 *
		 * any subset is removed before, subset of parameter sf is possible
		 *
		 * @param sf string features
		 * @param start start
		 * @param p_order order
		 * @param gap gap
		 * @param rev reverse
		 * @return if obtaining was successful
		 */
		inline bool obtain_from_char(CStringFeatures<char>* sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
		{
			return obtain_from_char_features(sf, start, p_order, gap, rev);
		}

		/** template obtain from char features
		 *
		 * any subset is removed before, subset of parameter sf is possible
		 *
		 * @param sf string features
		 * @param start start
		 * @param p_order order
		 * @param gap gap
		 * @param rev reverse
		 * @return if obtaining was successful
		 */
		template <class CT>
			bool obtain_from_char_features(CStringFeatures<CT>* sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
			{
				remove_subset();
				ASSERT(sf);

				CAlphabet* alpha=sf->get_alphabet();
				ASSERT(alpha->get_num_symbols_in_histogram() > 0);

				this->order=p_order;
				cleanup();

				num_vectors=sf->get_num_vectors();
				ASSERT(num_vectors>0);
				max_string_length=sf->get_max_vector_length()-start;
				features=new SGString<ST>[num_vectors];

				SG_DEBUG( "%1.0llf symbols in StringFeatures<*> %d symbols in histogram\n", sf->get_num_symbols(),
						alpha->get_num_symbols_in_histogram());

				for (int32_t i=0; i<num_vectors; i++)
				{
					int32_t len=-1;
					bool vfree;
					CT* c=sf->get_feature_vector(i, len, vfree);
					ASSERT(!vfree); // won't work when preprocessors are attached

					features[i].string=new ST[len];
					features[i].length=len;

					ST* str=features[i].string;
					for (int32_t j=0; j<len; j++)
						str[j]=(ST) alpha->remap_to_bin(c[j]);
				}

				original_num_symbols=alpha->get_num_symbols();
				int32_t max_val=alpha->get_num_bits();

				SG_UNREF(alpha);

				if (p_order>1)
					num_symbols=CMath::powl((floatmax_t) 2, (floatmax_t) max_val*p_order);
				else
					num_symbols=original_num_symbols;
				SG_INFO( "max_val (bit): %d order: %d -> results in num_symbols: %.0Lf\n", max_val, p_order, num_symbols);

				if ( ((floatmax_t) num_symbols) > CMath::powl(((floatmax_t) 2),((floatmax_t) sizeof(ST)*8)) )
				{
					SG_ERROR( "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
					return false;
				}

				SG_DEBUG( "translate: start=%i order=%i gap=%i(size:%i)\n", start, p_order, gap, sizeof(ST)) ;
				for (int32_t line=0; line<num_vectors; line++)
				{
					int32_t len=0;
					bool vfree;
					ST* fv=get_feature_vector(line, len, vfree);
					ASSERT(!vfree); // won't work when preprocessors are attached

					if (rev)
						CAlphabet::translate_from_single_order_reversed(fv, len, start+gap, p_order+gap, max_val, gap);
					else
						CAlphabet::translate_from_single_order(fv, len, start+gap, p_order+gap, max_val, gap);

					/* fix the length of the string -- hacky */
					features[line].length-=start+gap ;
					if (features[line].length<0)
						features[line].length=0 ;
				}

				compute_symbol_mask_table(max_val);

				return true;
			}

		/** check if length of each vector in this feature object equals the
		 * given length. if existant, only subset is checked
		 *
		 * possible for subset
		 *
		 * @param len vector length to check against
		 * @return if length of each vector in this feature object equals the
		 * given length.
		 */
		bool have_same_length(int32_t len=-1)
		{
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

		/** embed string features in bit representation in-place
		 *
		 * not implemented for subset
		 *
		 */
		inline void embed_features(int32_t p_order)
		{
			if (m_subset)
				SG_NOTIMPLEMENTED;

			ASSERT(alphabet->get_num_symbols_in_histogram() > 0);

			order=p_order;
			original_num_symbols=alphabet->get_num_symbols();
			int32_t max_val=alphabet->get_num_bits();

			if (p_order>1)
				num_symbols=CMath::powl((floatmax_t) 2, (floatmax_t) max_val*p_order);
			else
				num_symbols=original_num_symbols;

			SG_INFO( "max_val (bit): %d order: %d -> results in num_symbols: %.0Lf\n", max_val, p_order, num_symbols);

			if ( ((floatmax_t) num_symbols) > CMath::powl(((floatmax_t) 2),((floatmax_t) sizeof(ST)*8)) )
				SG_WARNING("symbols did not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);

			ST mask=0;
			for (int32_t i=0; i<p_order*max_val; i++)
				mask= (mask<<1) | ((ST) 1);

			for (int32_t i=0; i<num_vectors; i++)
			{
				int32_t len=features[i].length;

				if (len < p_order)
					SG_ERROR("Sequence must be longer than order (%d vs. %d)\n", len, p_order);

				ST* str=features[i].string;

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

				features[i].length=len-p_order+1;
			}

			compute_symbol_mask_table(max_val);
		}

		/** compute symbol mask table
		 *
		 * required to access bit-based symbols
		 *
		 * not implemented for subset
		 */
		inline void compute_symbol_mask_table(int64_t max_val)
		{
			if (m_subset)
				SG_NOTIMPLEMENTED;

			delete[] symbol_mask_table;
			symbol_mask_table=new ST[256];

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

		/** remap bit-based word to character sequence
		 *
		 * @param word word to remap
		 * @param seq sequence of size len that remapped characters are written to
		 * @param len length of sequence and word
		 */
		inline void unembed_word(ST word, uint8_t* seq, int32_t len)
		{
			uint32_t nbits= (uint32_t) alphabet->get_num_bits();

			ST mask=0;
			for (int32_t i=0; i<nbits; i++)
				mask=(mask<<1) | (ST) 1;

			for (int32_t i=0; i<len; i++)
			{
				ST w=(word & mask);
				seq[len-i-1]=alphabet->remap_to_char((uint8_t) w);
				word>>=nbits;
			}
		}

		/** embed a single word
		 *
		 * @param seq sequence of size len in a bitfield
		 * @param len
		 */
		inline ST embed_word(ST* seq, int32_t len)
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

		/** determine new maximum string length
		 *
		 * possible with subset
		 */
		void determine_maximum_string_length()
		{
			max_string_length=0;
			index_t num_str=get_num_vectors();
			for (int32_t i=0; i<num_str; i++)
			{
				max_string_length=CMath::max(max_string_length,
					features[subset_idx_conversion(i)].length);
			}
		}

		/** get a zero terminated copy of the string
		 *
		 * @param str the string to copy
		 * @return zero terminated copy of str
		 *
		 * note that this function is only sensible for character strings
		 */
		static ST* get_zero_terminated_string_copy(SGString<ST> str)
		{
			int32_t l=str.length;
			ST* s=new ST[l+1];
			memcpy(s, str.string, sizeof(ST)*l);
			s[l]='\0';
			return s;
		}

		/** set feature vector for sample num
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @param string string with the feature vector's content
		 * @param len length of the string
		 */
		virtual void set_feature_vector(int32_t num, ST* string, int32_t len)
		{
			ASSERT(features);
			ASSERT(num<get_num_vectors());

			int32_t real_num=subset_idx_conversion(num);


			features[real_num].length=len ;
			features[real_num].string=string ;

			max_string_length=CMath::max(len, max_string_length);
		}


		/** compute histogram over strings
		 *
		 * possible with subset
		 */
		virtual void get_histogram(float64_t** hist, int32_t* rows, int32_t* cols, bool normalize=true)
		{
			int32_t nsym=get_num_symbols();
			int32_t slen=get_max_vector_length();
			int64_t sz=int64_t(nsym)*slen*sizeof(float64_t);
			float64_t* h= (float64_t*) SG_MALLOC(sz);
			memset(h, 0, sz);

			float64_t* h_normalizer=new float64_t[slen];
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
			delete[] h_normalizer;

			*hist=h;
			*rows=nsym;
			*cols=slen;
		}

		/** create some random strings based on normalized histogram
		 *
		 * not possible with subset
		 */
		virtual void create_random(float64_t* hist, int32_t rows, int32_t cols, int32_t num_vec)
		{
			ASSERT(rows == get_num_symbols());
			cleanup();
			float64_t* randoms=new float64_t[cols];
			SGString<ST>* sf=new SGString<ST>[num_vec];

			for (int32_t i=0; i<num_vec; i++)
			{
				sf[i].string=new ST[cols];
				sf[i].length=cols;

				CMath::random_vector(randoms, cols, 0.0, 1.0);

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
					sf[i].string[j]=alphabet->remap_to_char(c);
				}
			}
			delete[] randoms;
			set_features(sf, num_vec, cols);
		}

		/*
		CStringFeatures<SSKTripleFeature>* obtain_sssk_triple_from_cha(int d1, int d2)
		{
			int *s;
			int32_t nStr=get_num_vectors();

			int32_t nfeat=0;
			for (int32_t i=0; i < nStr; ++i)
				nfeat += get_vector_length[i] - d1 -d2;
			SGString<SSKFeature>* F= new SGString<SSKFeature>[nfeat];
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
			ASSERT(nfeat==c);
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

		/** @return object name */
		inline virtual const char* get_name() const { return "StringFeatures"; }

		/** post method when subset is changed */
		virtual void subset_changed_post()
		{
			/* max string length has to be updated */
			determine_maximum_string_length();
		}
	protected:

		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * possible with subset
		 *
		 * @param num which vector
		 * @param len length of vector
		 * @return feature vector
		 */
		virtual ST* compute_feature_vector(int32_t num, int32_t& len)
		{
			ASSERT(features && num<get_num_vectors());

			int32_t real_num=subset_idx_conversion(num);

			len=features[real_num].length;
			if (len<=0)
				return NULL;

			ST* target=new ST[len];
			memcpy(target, features[real_num].string, len*sizeof(ST));
			return target;
		}

	private:
		void init()
		{
			set_generic<ST>();

			alphabet=NULL;
			num_vectors=0;
			features=NULL;
			single_string=NULL;
			length_of_single_string=0;
			max_string_length=0;
			order=0;
			symbol_mask_table=0;
			preprocess_on_get=false;
			feature_cache=NULL;

			m_parameters->add((CSGObject**) &alphabet, "alphabet");
			m_parameters->add_vector(&features, &num_vectors, "features",
					"This contains the array of features.");
			m_parameters->add_vector(&single_string,
					&length_of_single_string,
					"single_string",
					"Created by sliding window.");
			m_parameters->add(&max_string_length, "max_string_length",
					"Length of longest string.");
			m_parameters->add(&num_symbols, "num_symbols",
					"Number of used symbols.");
			m_parameters->add(&original_num_symbols, "original_num_symbols",
					"Original number of used symbols.");
			m_parameters->add(&order, "order",
					"Order used in higher order mapping.");
			m_parameters->add(&preprocess_on_get, "preprocess_on_get",
					"Preprocess on-the-fly?");

			/* TODO M_PARAMETERS->ADD?
			 * /// order used in higher order mapping
			 * ST* symbol_mask_table;
			 */
		}


	protected:

		/// alphabet
		CAlphabet* alphabet;

		/* number of string vectors (for subset, is updated) */
		int32_t num_vectors;

		/// this contains the array of features.
		SGString<ST>* features;

		/// true when single string / created by sliding window
		ST* single_string;

		/// length of prior single string
		int32_t length_of_single_string;

		/** length of longest string (for subset, is updated) */
		int32_t max_string_length;

		/// number of used symbols
		floatmax_t num_symbols;

		/// original number of used symbols (before higher order mapping)
		floatmax_t original_num_symbols;

		/// order used in higher order mapping
		int32_t order;

		/// order used in higher order mapping
		ST* symbol_mask_table;

		/// preprocess on-the-fly?
		bool preprocess_on_get;

		/** feature cache */
		CCache<ST>* feature_cache;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** get feature type the char feature can deal with
 *
 * @return feature type char
 */
template<> inline EFeatureType CStringFeatures<bool>::get_feature_type()
{
	return F_BOOL;
}

/** get feature type the char feature can deal with
 *
 * @return feature type char
 */
template<> inline EFeatureType CStringFeatures<char>::get_feature_type()
{
	return F_CHAR;
}

/** get feature type the BYTE feature can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CStringFeatures<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

/** get feature type the SHORT feature can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CStringFeatures<int16_t>::get_feature_type()
{
	return F_SHORT;
}

/** get feature type the WORD feature can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CStringFeatures<uint16_t>::get_feature_type()
{
	return F_WORD;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CStringFeatures<int32_t>::get_feature_type()
{
	return F_INT;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CStringFeatures<uint32_t>::get_feature_type()
{
	return F_UINT;
}

/** get feature type the LONG feature can deal with
 *
 * @return feature type LONG
 */
template<> inline EFeatureType CStringFeatures<int64_t>::get_feature_type()
{
	return F_LONG;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CStringFeatures<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

/** get feature type the SHORTREAL feature can deal with
 *
 * @return feature type SHORTREAL
 */
template<> inline EFeatureType CStringFeatures<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CStringFeatures<float64_t>::get_feature_type()
{
	return F_DREAL;
}

/** get feature type the LONGREAL feature can deal with
 *
 * @return feature type LONGREAL
 */
template<> inline EFeatureType CStringFeatures<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<> inline bool CStringFeatures<bool>::get_masked_symbols(bool symbol, uint8_t mask)
{
	return symbol;
}
template<> inline float32_t CStringFeatures<float32_t>::get_masked_symbols(float32_t symbol, uint8_t mask)
{
	return symbol;
}
template<> inline float64_t CStringFeatures<float64_t>::get_masked_symbols(float64_t symbol, uint8_t mask)
{
	return symbol;
}
template<> inline floatmax_t CStringFeatures<floatmax_t>::get_masked_symbols(floatmax_t symbol, uint8_t mask)
{
	return symbol;
}

template<> inline bool CStringFeatures<bool>::shift_offset(bool symbol, int32_t amount)
{
	return false;
}
template<> inline float32_t CStringFeatures<float32_t>::shift_offset(float32_t symbol, int32_t amount)
{
	return 0;
}
template<> inline float64_t CStringFeatures<float64_t>::shift_offset(float64_t symbol, int32_t amount)
{
	return 0;
}
template<> inline floatmax_t CStringFeatures<floatmax_t>::shift_offset(floatmax_t symbol, int32_t amount)
{
	return 0;
}

template<> inline bool CStringFeatures<bool>::shift_symbol(bool symbol, int32_t amount)
{
	return symbol;
}
template<> inline float32_t CStringFeatures<float32_t>::shift_symbol(float32_t symbol, int32_t amount)
{
	return symbol;
}
template<> inline float64_t CStringFeatures<float64_t>::shift_symbol(float64_t symbol, int32_t amount)
{
	return symbol;
}
template<> inline floatmax_t CStringFeatures<floatmax_t>::shift_symbol(floatmax_t symbol, int32_t amount)
{
	return symbol;
}

#ifndef SUNOS
template<> 	template <class CT> bool CStringFeatures<float32_t>::obtain_from_char_features(CStringFeatures<CT>* sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
template<> 	template <class CT> bool CStringFeatures<float64_t>::obtain_from_char_features(CStringFeatures<CT>* sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
template<> 	template <class CT> bool CStringFeatures<floatmax_t>::obtain_from_char_features(CStringFeatures<CT>* sf, int32_t start, int32_t p_order, int32_t gap, bool rev)
{
	return false;
}
#endif

template<> 	inline void CStringFeatures<float32_t>::embed_features(int32_t p_order)
{
}
template<> 	inline void CStringFeatures<float64_t>::embed_features(int32_t p_order)
{
}
template<> 	inline void CStringFeatures<floatmax_t>::embed_features(int32_t p_order)
{
}

template<> 	inline void CStringFeatures<float32_t>::compute_symbol_mask_table(int64_t max_val)
{
}
template<> 	inline void CStringFeatures<float64_t>::compute_symbol_mask_table(int64_t max_val)
{
}
template<> 	inline void CStringFeatures<floatmax_t>::compute_symbol_mask_table(int64_t max_val)
{
}

template<> 	inline float32_t CStringFeatures<float32_t>::embed_word(float32_t* seq, int32_t len)
{
	return 0;
}
template<> 	inline float64_t CStringFeatures<float64_t>::embed_word(float64_t* seq, int32_t len)
{
	return 0;
}
template<> 	inline floatmax_t CStringFeatures<floatmax_t>::embed_word(floatmax_t* seq, int32_t len)
{
	return 0;
}

template<> 	inline void CStringFeatures<float32_t>::unembed_word(float32_t word, uint8_t* seq, int32_t len)
{
}
template<> 	inline void CStringFeatures<float64_t>::unembed_word(float64_t word, uint8_t* seq, int32_t len)
{
}
template<> 	inline void CStringFeatures<floatmax_t>::unembed_word(floatmax_t word, uint8_t* seq, int32_t len)
{
}
#define LOAD(f_load, sg_type)												\
template<> inline void CStringFeatures<sg_type>::load(CFile* loader)		\
{																			\
	SG_INFO( "loading...\n");												\
																			\
	SG_SET_LOCALE_C;													\
	SGString<sg_type>* strs;												\
	int32_t num_str;														\
	int32_t max_len;														\
	loader->f_load(strs, num_str, max_len);									\
	set_features(strs, num_str, max_len);									\
	SG_RESET_LOCALE;													\
}

LOAD(get_string_list, bool)
LOAD(get_string_list, char)
LOAD(get_int8_string_list, int8_t)
LOAD(get_string_list, uint8_t)
LOAD(get_string_list, int16_t)
LOAD(get_string_list, uint16_t)
LOAD(get_string_list, int32_t)
LOAD(get_uint_string_list, uint32_t)
LOAD(get_long_string_list, int64_t)
LOAD(get_ulong_string_list, uint64_t)
LOAD(get_string_list, float32_t)
LOAD(get_string_list, float64_t)
LOAD(get_longreal_string_list, floatmax_t)
#undef LOAD

#define SAVE(f_write, sg_type)												\
template<> inline void CStringFeatures<sg_type>::save(CFile* writer)		\
{ 																			\
	if (m_subset)															\
		SG_ERROR("save() is not possible on subset");						\
	SG_SET_LOCALE_C;													\
	ASSERT(writer);															\
	writer->f_write(features, num_vectors);									\
	SG_RESET_LOCALE;													\
}

SAVE(set_string_list, bool)
SAVE(set_string_list, char)
SAVE(set_int8_string_list, int8_t)
SAVE(set_string_list, uint8_t)
SAVE(set_string_list, int16_t)
SAVE(set_string_list, uint16_t)
SAVE(set_string_list, int32_t)
SAVE(set_uint_string_list, uint32_t)
SAVE(set_long_string_list, int64_t)
SAVE(set_ulong_string_list, uint64_t)
SAVE(set_string_list, float32_t)
SAVE(set_string_list, float64_t)
SAVE(set_longreal_string_list, floatmax_t)
#undef SAVE
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif // _CSTRINGFEATURES__H__
