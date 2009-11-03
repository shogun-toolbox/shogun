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

#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Cache.h"
#include "preproc/PreProc.h"
#include "preproc/StringPreProc.h"
#include "features/Features.h"
#include "features/Alphabet.h"
#include "lib/DynamicArray.h"
#include "lib/File.h"
#include "lib/MemoryMappedFile.h"
#include "lib/Mathematics.h"
#include "lib/Compressor.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace shogun
{
class CCompressor;
class CAlphabet;
enum EAlphabet;
template <class T> class CDynamicArray;
class CFile;
template <class T> class CMemoryMappedFile;
class CMath;
template <class ST> class CStringPreProc;
template <class T> class T_STRING;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** template class T_STRING */
template <class T> class T_STRING
{


#ifdef HAVE_BOOST_SERIALIZATION

  private:


  friend class ::boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int archive_version) const
  {

    //SG_DEBUG("archiving T_STRING\n");

  	ar & length;

    for (int i=0; i < length; ++i) {
      ar & string[i];
    }

    //SG_DEBUG("done archiving T_STRING\n");

  }

  template<class Archive>
  void load(Archive & ar, const unsigned int archive_version)
  {

	//SG_DEBUG("archiving T_STRING\n");

  	ar & length;

    string = new T[length];

    for (int i=0; i < length; ++i) {
      ar & string[i];
    }

    //SG_DEBUG("done archiving T_STRING\n");

  }

  GLOBAL_BOOST_SERIALIZATION_SPLIT_MEMBER();


#endif //HAVE_BOOST_SERIALIZATION

    public:
	/** string */
	T* string;
	/** length of string */
	int32_t length;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS


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
 */
template <class ST> class CStringFeatures : public CFeatures
{
	public:
		/** default constructor
		 *
		 */
        CStringFeatures() : CFeatures(0), alphabet(NULL), num_vectors(0),
		features(NULL), single_string(NULL),length_of_single_string(0),
        max_string_length(0), order(0), symbol_mask_table(NULL),
		preprocess_on_get(false), feature_cache(NULL)
        {
        }

		/** constructor
		 *
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(EAlphabet alpha)
		: CFeatures(0), num_vectors(0), features(NULL),
			single_string(NULL),length_of_single_string(0),
			max_string_length(0), order(0), symbol_mask_table(NULL),
			preprocess_on_get(false), feature_cache(NULL)
		{
			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
		}

		/** constructor
		 *
		 * @param p_features new features
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(T_STRING<ST>* p_features, int32_t p_num_vectors,
				int32_t p_max_string_length, EAlphabet alpha)
		: CFeatures(0), num_vectors(0), features(NULL),
			single_string(NULL),length_of_single_string(0),
			max_string_length(0), order(0), symbol_mask_table(NULL),
			preprocess_on_get(false), feature_cache(NULL)
		{
			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
			set_features(p_features, p_num_vectors, p_max_string_length);
		}

		/** constructor
		 *
		 * @param alpha alphabet to use for string features
		 */
		CStringFeatures(CAlphabet* alpha)
		: CFeatures(0), num_vectors(0), features(NULL),
			single_string(NULL),length_of_single_string(0),
			max_string_length(0), order(0), symbol_mask_table(NULL),
			preprocess_on_get(false), feature_cache(NULL)
		{
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
			ASSERT(orig.single_string == NULL); //not implemented

			alphabet=orig.alphabet;
			SG_REF(alphabet);

			if (orig.features)
			{
				features=new T_STRING<ST>[orig.num_vectors];

				for (int32_t i=0; i<num_vectors; i++)
				{
					features[i].string=new ST[orig.features[i].length];
					ASSERT(features[i].string);
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
		}

		/** constructor
		 *
		 * @param fname filename to load features from
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(char* fname, EAlphabet alpha=DNA)
		: CFeatures(fname), num_vectors(0), features(NULL), single_string(NULL),
			length_of_single_string(0), max_string_length(0), order(0),
			symbol_mask_table(NULL), preprocess_on_get(false), feature_cache(NULL)
		{
			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
			load(fname);
		}

		virtual ~CStringFeatures()
		{
			cleanup();

			SG_UNREF(alphabet);
		}

		/** cleanup string features */
		virtual void cleanup()
		{
			if (single_string)
			{
				delete[] single_string;
				single_string=NULL;
			}
			else
			{
				for (int32_t i=0; i<num_vectors; i++)
				{
					delete[] features[i].string;
					features[i].length=0;
				}
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
		 * @param dst destination where vector will be stored
		 * @param len number of features in vector
		 * @param num index of the string
		 */
		void get_feature_vector(ST** dst, int32_t* len, int32_t num)
		{
			ASSERT(features);
			if (num>=num_vectors)
			{
				SG_ERROR("Index out of bounds (number of strings %d, you "
						"requested %d)\n", num_vectors, num);
			}

			*len=features[num].length;
			*dst=(ST*) malloc(*len * sizeof(ST));
			memcpy(*dst, features[num].string, *len * sizeof(ST));
		}

		/** set string for selected example num
		 *
		 * @param src destination where vector will be stored
		 * @param len number of features in vector
		 * @param num index of the string
		 */
		void set_feature_vector(ST* src, int32_t len, int32_t num)
		{
			ASSERT(features);
			if (num>=num_vectors)
			{
				SG_ERROR("Index out of bounds (number of strings %d, you "
						"requested %d)\n", num_vectors, num);
			}

			if (len<=0)
				SG_ERROR("String has zero or negative length\n");


			delete[] features[num].string;
			features[num].length=len;
			features[num].string=new ST[len];
			memcpy(features[num].string, src, len*sizeof(ST));

			determine_maximum_string_length();
		}

		/** get feature vector for sample num
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
			ASSERT(num<num_vectors);

			if (!preprocess_on_get)
			{

				dofree=false;
				len=features[num].length;
				return features[num].string;
			}
			else
			{
				SG_DEBUG( "computing feature vector!\n") ;

				ST* feat=NULL;
				dofree=false;

				if (feature_cache)
				{
					feat=feature_cache->lock_entry(num);

					if (feat)
						return feat;
					else
					{
						feat=feature_cache->set_entry(num);
					}
				}

				if (!feat)
					dofree=true;
				feat=compute_feature_vector(num, len, feat);

				if (get_num_preproc())
				{
					int32_t tmp_len=len;
					ST* tmp_feat_before = feat;
					ST* tmp_feat_after = NULL;

					for (int32_t i=0; i<get_num_preproc(); i++)
					{
						CStringPreProc<ST>* p = (CStringPreProc<ST>*) get_preproc(i);
						tmp_feat_after=p->apply_to_string(tmp_feat_before, tmp_len);
						SG_UNREF(p);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(ST)*tmp_len);
					delete[] tmp_feat_after;

					len=tmp_len;
				}
				return feat ;
			}
		}

		/** free feature vector
		 *
		 * @param feat_vec feature vector to free
		 * @param num index in feature cache
		 * @param dofree if vector should be really deleted
		 */
		void free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (dofree)
				delete[] feat_vec ;
		}

		/** get feature
		 *
		 * @param vec_num which vector
		 * @param feat_num which feature
		 * @return feature
		 */
		virtual ST inline get_feature(int32_t vec_num, int32_t feat_num)
		{
			ASSERT(features && vec_num<num_vectors);
			ASSERT(feat_num<features[vec_num].length);

			return features[vec_num].string[feat_num];
		}

		/** get vector length
		 *
		 * @param vec_num which vector
		 * @return length of vector
		 */
		virtual inline int32_t get_vector_length(int32_t vec_num)
		{
			ASSERT(features && vec_num<num_vectors);
			return features[vec_num].length;
		}

		/** get maximum vector length
		 *
		 * @return maximum vector/string length
		 */
		virtual inline int32_t get_max_vector_length()
		{
			return max_string_length;
		}

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		virtual inline int32_t get_num_vectors() { return num_vectors; }

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
		 * @param fname filename to load from
		 * @return if loading was successful
		 */
		virtual bool load(char* fname)
		{
			SG_INFO( "loading...\n");
			int64_t length=0;
			max_string_length=0;

			CFile f(fname, 'r', F_CHAR);
			char* feature_matrix=f.load_char_data(NULL, length);

			SG_DEBUG("char data now at %p of length %ld\n", 
					feature_matrix, (int64_t) length);

			num_vectors=0;

			if (f.is_ok())
			{
				for (int64_t i=0; i<length; i++)
				{
					if (feature_matrix[i]=='\n')
						num_vectors++;
				}

				SG_INFO( "file contains %ld vectors\n", num_vectors);
				features= new T_STRING<ST>[num_vectors];

				int64_t index=0;
				for (int32_t lines=0; lines<num_vectors; lines++)
				{
					char* p=&feature_matrix[index];
					int32_t columns=0;

					for (columns=0; index+columns<length && p[columns]!='\n'; columns++);

					if (index+columns>=length && p[columns]!='\n') {
						SG_ERROR( "error in \"%s\":%d\n", fname, lines);
					}

					features[lines].length=columns;
					features[lines].string=new ST[columns];

					max_string_length=CMath::max(max_string_length,columns);

					for (int32_t i=0; i<columns; i++)
						features[lines].string[i]= ((ST) p[i]);

					index+= features[lines].length+1;
				}

				num_symbols=4; //FIXME
				return true;
			}
			else
				SG_ERROR( "reading file failed\n");

			return false;
		}

		/** load DNA features from file
		 *
		 * @param fname filename to load from
		 * @param remap_to_bin if remap_to_bin
		 * @return if loading was successful
		 */
		bool load_dna_file(char* fname, bool remap_to_bin=true)
		{
			bool result=false;

			size_t blocksize=1024*1024;
			size_t required_blocksize=0;
			uint8_t* dummy=new uint8_t[blocksize];
			uint8_t* overflow=NULL;
			int32_t overflow_len=0;

			num_symbols=4;
			cleanup();

			CAlphabet* alpha=new CAlphabet(DNA);
			CAlphabet* alpha_bin=new CAlphabet(RAWDNA);

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
				dummy = new uint8_t[blocksize];
				overflow = new uint8_t[blocksize];
				features=new T_STRING<ST>[num_vectors];

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
								alpha_bin->add_string_to_histogram(features[lines].string, features[lines].length);
							}
							else
							{
								for (int32_t j=0; j<overflow_len; j++)
									features[lines].string[j]=overflow[j];
								for (int32_t j=0; j<len; j++)
									features[lines].string[j+overflow_len]=dummy[old_sz+j];
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
				result=true;
				SG_INFO("file successfully read\n");
				SG_INFO("max_string_length=%d\n", max_string_length);
				SG_INFO("num_strings=%d\n", num_vectors);
			}

			fclose(f);
			delete[] dummy;

			SG_UNREF(alphabet);

			if (remap_to_bin)
				alphabet = alpha_bin;
			else
				alphabet = alpha;
			SG_REF(alphabet);

			return result;
		}

		/** load fasta file as string features
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @return if loading was successful
		 */
		bool load_fasta_file(const char* fname, bool ignore_invalid=false)
		{
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

			T_STRING<ST>* strings=new T_STRING<ST>[num];
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

						len = fasta_len-spanned_lines;
						strings[i].string=new ST[len];
						strings[i].length=len;

						ST* str=strings[i].string;
						int32_t idx=0;
						SG_DEBUG("'%.*s', len=%d, spanned_lines=%d\n", (int32_t) id_len, id, (int32_t) len, (int32_t) spanned_lines);

						for (int32_t j=0; j<fasta_len; j++)
						{
							if (fasta[j]=='\n')
								continue;

							ST c = (ST) fasta[j];

							if (ignore_invalid  && !alphabet->is_valid((uint8_t) fasta[j]))
								c = (ST) 'A';

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
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @param bitremap_in_single_string if set to true, do binary embedding of symbols
		 * @return if loading was successful
		 */
		bool load_fastq_file(const char* fname,
				bool ignore_invalid=false, bool bitremap_in_single_string=false)
		{
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

			T_STRING<ST>* strings;

			ST* str;
			if (bitremap_in_single_string)
			{
				strings=new T_STRING<ST>[1];
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
				strings=new T_STRING<ST>[num];

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
		 * @param dirname directory name to load from
		 * @return if loading was successful
		 */
		bool load_from_directory(char* dirname)
		{
			struct dirent **namelist;
			int32_t n;

            CIO::set_dirname(dirname);

			SG_DEBUG("dirname '%s'\n", dirname);

			n = scandir(dirname, &namelist, &CIO::filter, alphasort);
			if (n <= 0)
			{
				SG_ERROR("error calling scandir - no files found\n");
				return false;
			}
			else
			{
				T_STRING<ST>* strings=NULL;

				int32_t num=0;
				int32_t max_len=-1;

				//usually n==num_vec, but it might not in race conditions
				//(file perms modified, file erased)
				strings=new T_STRING<ST>[n];

				for (int32_t i=0; i<n; i++)
				{
					char* fname=CIO::concat_filename(namelist[i]->d_name);

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
							fread(str, sizeof(ST), filesize, f);
							strings[num].string=str;
							strings[num].length=filesize;
							max_len=CMath::max(max_len, strings[num].length);

							num++;
							fclose(f);
						}
					}
					else
						SG_ERROR("empty or non readable file \'%s\'\n", fname);

					free(namelist[i]);
				}
				free(namelist);

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
		 * @param p_features new features
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 * @return if setting was successful
		 */
		bool set_features(T_STRING<ST>* p_features, int32_t p_num_vectors, int32_t p_max_string_length)
		{
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

					this->features=p_features;
					this->num_vectors=p_num_vectors;
					this->max_string_length=p_max_string_length;

					return true;
				}
				else
					SG_UNREF(alpha);
			}

			return false;
		}

		/** get_features
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual T_STRING<ST>* get_features(int32_t& num_str, int32_t& max_str_len)
		{
			num_str=num_vectors;
			max_str_len=max_string_length;
			return features;
		}

		/** get_features  (swig compatible)
		 *
		 * @param dst string features (returned)
		 * @param num_str number of strings (returned)
		 */
		virtual void get_features(T_STRING<ST>** dst, int32_t* num_str)
		{
			*num_str=num_vectors;
			*dst=features;
		}

		/** save features to file
		 *
		 * @param dest filename to save to
		 * @return if saving was successful
		 */
		virtual bool save(char* dest)
		{
			return false;
		}

		/** load compressed features from file
		 *
		 * @param src filename to load from
		 * @param decompress whether to decompress on loading
		 * @return if loading was successful
		 */
		virtual bool load_compressed(char* src, bool decompress)
		{
			FILE* file=NULL;

			if (!(file=fopen(src, "r")))
				return false;
			cleanup();

			// header shogun v0
			char id[4];
			fread(&id[0], sizeof(char), 1, file);
			ASSERT(id[0]=='S');
			fread(&id[1], sizeof(char), 1, file);
			ASSERT(id[1]=='G');
			fread(&id[2], sizeof(char), 1, file);
			ASSERT(id[2]=='V');
			fread(&id[3], sizeof(char), 1, file);
			ASSERT(id[3]=='0');

			//compression type
			uint8_t c;
			fread(&c, sizeof(uint8_t), 1, file);
			CCompressor* compressor= new CCompressor((CCompressor::E_COMPRESSION_TYPE) c);
			//alphabet
			uint8_t a;
			delete alphabet;
			fread(&a, sizeof(uint8_t), 1, file);
			alphabet=new CAlphabet((EAlphabet) a);
			// number of vectors
			fread(&num_vectors, sizeof(int32_t), 1, file);
			ASSERT(num_vectors>0);
			// maximum string length
			fread(&max_string_length, sizeof(int32_t), 1, file);
			ASSERT(max_string_length>0);

			features=new T_STRING<ST>[num_vectors];

			// vectors
			for (int32_t i=0; i<num_vectors; i++)
			{
				// vector len compressed
				int32_t len_compressed;
				fread(&len_compressed, sizeof(int32_t), 1, file);
				// vector len uncompressed
				int32_t len_uncompressed;
				fread(&len_uncompressed, sizeof(int32_t), 1, file);

				// vector raw data
				if (decompress)
				{
					//ST* compressed=NULL;
					//fread(compressed, len_compressed, 1, file);

					//uint64_t uncompressed_size=len_uncompressed;
					//compressor->decompress(compressed, len_compressed,
					//		uncompressed, uncompressed_size);
					//ASSERT(compressed);
					//features[i].string=new ST[
					//features[i].length=len_compressed;
				}
				else
				{
					//features[i].string=new ST[
					//features[i].length=len_compressed;
				}
			}

			delete compressor;
			fclose(file);
			return false;
		}

		/** save compressed features to file
		 *
		 * @param dest filename to save to
		 * @param compression compressor to use
		 * @param compression level to use (1-9)
		 * @return if saving was successful
		 */
		virtual bool save_compressed(char* dest, CCompressor::E_COMPRESSION_TYPE compression, int level)
		{
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

				compressor->compress((uint8_t*) vec, (uint64_t) sizeof(ST)*len,
						compressed, compressed_size, level);
				ASSERT(compressed);

				int32_t len_compressed = (int32_t) compressed_size;
				// vector len compressed
				fwrite(&len_compressed, sizeof(int32_t), 1, file);
				// vector len uncompressed
				fwrite(&len, sizeof(int32_t), 1, file);
				// vector raw data
				fwrite(compressed, compressed_size, 1, file);

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
		virtual bool apply_preproc(bool force_preprocessing=false)
		{
			SG_DEBUG( "force: %d\n", force_preprocessing);

			for (int32_t i=0; i<get_num_preproc(); i++)
			{
				if ( (!is_preprocessed(i) || force_preprocessing) )
				{
					set_preprocessed(i);
					CStringPreProc<ST>* p = (CStringPreProc<ST>*) get_preproc(i);
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
		 * @param window_size window size
		 * @param step_size step size
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_sliding_window(int32_t window_size, int32_t step_size, int32_t skip=0)
		{
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

			T_STRING<ST>* f=new T_STRING<ST>[num_vectors];
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
		 * @param window_size window size
		 * @param positions positions
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_position_list(int32_t window_size, CDynamicArray<int32_t>* positions, int32_t skip=0)
		{
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

			T_STRING<ST>* f=new T_STRING<ST>[num_vectors];
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
				ASSERT(sf);

				CAlphabet* alpha=sf->get_alphabet();
				ASSERT(alpha->get_num_symbols_in_histogram() > 0);

				this->order=p_order;
				cleanup();

				num_vectors=sf->get_num_vectors();
				ASSERT(num_vectors>0);
				max_string_length=sf->get_max_vector_length()-start;
				features=new T_STRING<ST>[num_vectors];

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
		 * given length.
		 *
		 * @param len vector length to check against
		 * @return if length of each vector in this feature object equals the
		 * given length.
		 */
		bool have_same_length(int32_t len=-1)
		{
			if (len!=-1)
			{
				if (len!=get_max_vector_length())
					return false;
			}
			len = get_max_vector_length();

			for (int32_t i=0; i<num_vectors; i++)
			{
				if (get_vector_length(i)!=len)
					return false;
			}

			return true;
		}

		/** embed string features in bit representation in-place
		 *
		 *
		 */
		inline void embed_features(int32_t p_order)
		{
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

				ST* str = features[i].string;

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
		 */
		inline void compute_symbol_mask_table(int64_t max_val)
		{
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
		 */
		void determine_maximum_string_length()
		{
			max_string_length=0;

			for (int32_t i=0; i<num_vectors; i++)
				max_string_length=CMath::max(max_string_length, features[i].length);
		}

		static ST* get_zero_terminated_string_copy(T_STRING<ST> str)
		{
			int32_t l=str.length;
			ST* s=new ST[l+1];
			memcpy(s, str.string, sizeof(ST)*l);
			s[l]='\0';
			return s;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "StringFeatures"; }

	protected:

		/** set feature vector for sample num
		 *
		 * @param num index of feature vector
		 * @param string string with the feature vector's content
		 * @param len length of the string
		 */
		virtual void set_feature_vector(int32_t num, ST* string, int32_t len)
		{
			ASSERT(features);
			ASSERT(num<num_vectors);

			features[num].length=len ;
			features[num].string=string ;
		}

		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * default implementation returns
		 *
		 * @param num num
		 * @param len len
		 * @param target
		 * @return feature vector
		 */
		virtual ST* compute_feature_vector(int32_t num, int32_t& len, ST* target=NULL)
		{
			ASSERT(features && num<num_vectors);

			len=features[num].length;
			if (len && !target) 
				target=new ST[len];
			memcpy(target, features[num].string, len*sizeof(ST));
			return target;
		}

#ifdef HAVE_BOOST_SERIALIZATION
    private:

        friend class ::boost::serialization::access;
        template<class Archive>
            void save(Archive & ar, const unsigned int archive_version) const
            {

				SG_DEBUG("archiving StringFeatures\n");

                ar & ::boost::serialization::base_object<CFeatures>(*this);

                ar & alphabet;

                ar & num_vectors;
                for (int i=0; i < num_vectors; ++i) {
                    ar & features[i];
                }

                ar & length_of_single_string;
                for (int i=0; i < length_of_single_string; ++i) {
                    ar & single_string[i];
                }

                ar & max_string_length;
                ar & num_symbols;
                ar & original_num_symbols;
                ar & order;

                /// order used in higher order mapping
                //TODO?! how long
                //ST* symbol_mask_table;

                SG_DEBUG("done archiving StringFeatures\n");

            }

        template<class Archive>
            void load(Archive & ar, const unsigned int archive_version)
            {

				SG_DEBUG("archiving StringFeatures\n");

                ar & ::boost::serialization::base_object<CFeatures>(*this);


                ar & alphabet;

                ar & num_vectors;

                //T_STRING<ST>* features = new T_STRING<ST>[num_vectors];
                features = new T_STRING<ST>[num_vectors];
                for (int i=0; i < num_vectors; ++i) {
                    ar & features[i];
                }


                ar & length_of_single_string;

                //ST* single_string = new ST[length_of_single_string];
                single_string = new ST[length_of_single_string];
                for (int i=0; i < length_of_single_string; ++i) {
                    ar & single_string[i];
                }

                ar & max_string_length;
                ar & num_symbols;
                ar & original_num_symbols;
                ar & order;

                /// order used in higher order mapping
                //TODO?! how long -> num_of_symbols?
                //ST* symbol_mask_table;

                SG_DEBUG("done archiving StringFeatures\n");

            }

        GLOBAL_BOOST_SERIALIZATION_SPLIT_MEMBER();


#endif //HAVE_BOOST_SERIALIZATION


	protected:

		/// alphabet
		CAlphabet* alphabet;

		/// number of string vectors
		int32_t num_vectors;

		/// this contains the array of features.
		T_STRING<ST>* features;

		/// true when single string / created by sliding window
		ST* single_string;

		/// length of prior single string
		int32_t length_of_single_string;

		/// length of longest string
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
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif // _CSTRINGFEATURES__H__
