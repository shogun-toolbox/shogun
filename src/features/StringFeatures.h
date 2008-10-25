/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__


#include "preproc/PreProc.h"
#include "preproc/StringPreProc.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "features/Alphabet.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/DynamicArray.h"
#include "lib/File.h"
#include "lib/Mathematics.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

class CFile;

template <class ST> class CStringPreProc;

/** template class T_STRING */
template <class T> struct T_STRING
{
	/** string */
	T* string;
	/** length of string */
	INT length;
};

template <class T> char* get_zero_terminated_string_copy(T_STRING<T> str)
{
	INT l=str.length;
	char* s=new char[l+1];
	memcpy(s, str.string, sizeof(char)*l);
	s[l]='\0';
	return s;
}

/** Template class StringFeatures implements a list of strings. As this class
 * is template the underlying storage type is quite arbitrary and not limited
 * to character strings, but could also be sequences of floating point numbers
 * etc. Strings differ from matrices (cf. CSimpleFeatures) in a way that the
 * dimensionality of the feature vectors (i.e. the strings) is not fixed; it
 * may vary between strings.
 * 
 * Most string kernels require StringFeatures but a number of them actually
 * requires strings to have same length.
 *
 * Note: StringFeatures do not support PreProcs
 */
template <class ST> class CStringFeatures : public CFeatures
{
	public:
		/** constructor
		 *
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(EAlphabet alpha)
		: CFeatures(0), num_vectors(0), features(NULL),
			single_string(NULL),length_of_single_string(0),
			max_string_length(0), order(0), selected_vector(0),
			symbol_mask_table(NULL)
		{
			alphabet=new CAlphabet(alpha);
			SG_REF(alphabet);
			num_symbols=alphabet->get_num_symbols();
			original_num_symbols=num_symbols;
		}

		/** constructor
		 *
		 * @param alpha alphabet to use for string features
		 */
		CStringFeatures(CAlphabet* alpha)
		: CFeatures(0), num_vectors(0), features(NULL),
			single_string(NULL),length_of_single_string(0),
			max_string_length(0), order(0), selected_vector(0),
			symbol_mask_table(NULL)
	{
		ASSERT(alpha);
		alphabet=new CAlphabet(alpha);
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
			order(orig.order), selected_vector(orig.selected_vector)
		{
			ASSERT(orig.single_string == NULL); //not implemented

			alphabet=new CAlphabet(orig.alphabet);
			SG_REF(alphabet);

			if (orig.features)
			{
				features=new T_STRING<ST>[orig.num_vectors];

				for (INT i=0; i<num_vectors; i++)
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
				for (INT i=0; i<256; i++)
					symbol_mask_table[i]=orig.symbol_mask_table[i];
			}
		}

		/** constructor
		 *
		 * @param fname filename to load features from
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(char* fname, EAlphabet alpha=DNA)
		: CFeatures(fname), num_vectors(0),
			features(NULL), single_string(NULL),
			length_of_single_string(0), max_string_length(0),
			order(0), selected_vector(0), symbol_mask_table(NULL)
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

#ifdef HAVE_SWIG
			SG_UNREF(alphabet);
#else
			delete alphabet;
#endif
		}

		/** cleanup string features */
		void cleanup()
		{
			if (single_string)
			{
				delete[] single_string;
				single_string=NULL;
			}
			else
			{
				for (int i=0; i<num_vectors; i++)
				{
					delete[] features[i].string;
					features[i].length=0;
				}
			}
			num_vectors=0;
			delete[] features;

			delete[] symbol_mask_table;
			alphabet->clear_histogram();
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
		inline virtual EFeatureType get_feature_type();

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

		/** select feature vector
		 *
		 * @param num which feature vector to select
		 */
		void select_feature_vector(INT num)
		{
			ASSERT(features);
			ASSERT(num<num_vectors);

			selected_vector=num;
		}

		/** get feature vector for selected example
		 *
		 * @param dst destination where vector will be stored
		 * @param len number of features in vector
		 */
		void get_string(ST** dst, INT* len)
		{
			ASSERT(features);
			ASSERT(selected_vector<num_vectors);

			*len=features[selected_vector].length;
			*dst=new ST[*len];
			memcpy(*dst, features[selected_vector].string, *len * sizeof(ST));
		}

		/** get feature vector for sample num
		 *
		 * @param num index of feature vector
		 * @param len length is returned by reference
		 * @return feature vector for sample num
		 */
		virtual ST* get_feature_vector(INT num, INT& len)
		{
			ASSERT(features);
			ASSERT(num<num_vectors);

			len=features[num].length;
			return features[num].string;
		}

		/** set feature vector for sample num
		 *
		 * @param num index of feature vector
		 * @param string string with the feature vector's content
		 * @param len length of the string
		 */
		virtual void set_feature_vector(INT num, ST* string, INT len)
		{
			ASSERT(features);
			ASSERT(num<num_vectors);

			features[num].length=len ;
			features[num].string=string ;
		}

		/** get feature
		 *
		 * @param vec_num which vector
		 * @param feat_num which feature
		 * @return feature
		 */
		virtual ST inline get_feature(INT vec_num, INT feat_num)
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
		virtual inline INT get_vector_length(INT vec_num)
		{
			ASSERT(features && vec_num<num_vectors);
			return features[vec_num].length;
		}

		/** get maximum vector length
		 *
		 * @return maximum vector/string length
		 */
		virtual inline INT get_max_vector_length()
		{
			return max_string_length;
		}

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		virtual inline INT get_num_vectors() { return num_vectors; }

		/** get number of symbols
		 *
		 * Note: LONGREAL sounds weird, but LONG is not long enough
		 *
		 * @return number of symbols
		 */
		inline LONGREAL get_num_symbols() { return num_symbols; }

		/** get maximum number of symbols
		 *
		 * Note: LONGREAL sounds weird, but LONG is not long enough
		 *
		 * @return maximum number of symbols
		 */
		inline LONGREAL get_max_num_symbols() { return CMath::powl(2,sizeof(ST)*8); }

		// these functions are necessary to find out about a former conversion process

		/** number of symbols before higher order mapping
		 *
		 * @return original number of symbols
		 */
		inline LONGREAL get_original_num_symbols() { return original_num_symbols; }

		/** order used for higher order mapping
		 *
		 * @return order
		 */
		inline INT get_order() { return order; }

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
		inline ST shift_offset(ST offset, INT amount)
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
		inline ST shift_symbol(ST symbol, INT amount)
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
			LONG length=0;
			max_string_length=0;

			CFile f(fname, 'r', F_CHAR);
			char* feature_matrix=f.load_char_data(NULL, length);

			num_vectors=0;

			if (f.is_ok())
			{
				for (long i=0; i<length; i++)
				{
					if (feature_matrix[i]=='\n')
						num_vectors++;
				}

				SG_INFO( "file contains %ld vectors\n", num_vectors);
				features= new T_STRING<ST>[num_vectors];

				long index=0;
				for (INT lines=0; lines<num_vectors; lines++)
				{
					char* p=&feature_matrix[index];
					INT columns=0;

					for (columns=0; index+columns<length && p[columns]!='\n'; columns++);

					if (index+columns>=length && p[columns]!='\n') {
						SG_ERROR( "error in \"%s\":%d\n", fname, lines);
					}

					features[lines].length=columns;
					features[lines].string=new ST[columns];

					max_string_length=CMath::max(max_string_length,columns);

					for (INT i=0; i<columns; i++)
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
			INT overflow_len=0;

			num_symbols=4;
			cleanup();

			CAlphabet* alpha=new CAlphabet(DNA);

			FILE* f=fopen(fname, "ro");

			if (f)
			{
				num_vectors=0;
				max_string_length=0;

				SG_INFO("counting line numbers in file %s\n", fname);
				SG_DEBUG("block_size=%d\n", required_blocksize);
				size_t sz=blocksize;
				size_t block_offs=0;
				size_t old_block_offs=0;
				fseek(f, 0, SEEK_END);
				size_t fsize=ftell(f);
				rewind(f);

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
				INT lines=0;
				while (sz == blocksize)
				{
					sz=fread(dummy, sizeof(uint8_t), blocksize, f);

					size_t old_sz=0;
					for (size_t i=0; i<sz; i++)
					{
						if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
						{
							INT len=i-old_sz;
							//SG_PRINT("i:%d len:%d old_sz:%d\n", i, len, old_sz);
							max_string_length=CMath::max(max_string_length, len+overflow_len);

							features[lines].length=len;
							features[lines].string=new ST[len];

							if (remap_to_bin)
							{
								for (INT j=0; j<overflow_len; j++)
									features[lines].string[j]=alpha->remap_to_bin(overflow[j]);
								for (INT j=0; j<len; j++)
									features[lines].string[j+overflow_len]=alpha->remap_to_bin(dummy[old_sz+j]);
							}
							else
							{
								for (INT j=0; j<overflow_len; j++)
									features[lines].string[j]=overflow[j];
								for (INT j=0; j<len; j++)
									features[lines].string[j+overflow_len]=dummy[old_sz+j];
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
			delete alpha;
			delete[] dummy;

#ifdef HAVE_SWIG
			SG_UNREF(alphabet);
#else
			delete alphabet;
#endif
			if (remap_to_bin)
				alphabet = new CAlphabet(RAWDNA);
			else
				alphabet = new CAlphabet(DNA);
			SG_REF(alphabet);

			return result;
		}

		/** load features from directory
		 *
		 * @param dirname directory name to load from
		 * @return if loading was successful
		 */
		bool load_from_directory(char* dirname)
		{
			struct dirent **namelist;
			int n;

			io.set_dirname(dirname);

			n = scandir(dirname, &namelist, io.filter, alphasort);
			if (n <= 0)
			{
				SG_ERROR( "error calling scandir\n");
				return false;
			}
			else
			{
				T_STRING<ST>* strings=NULL;
				alphabet->clear_histogram();

				INT num=0;
				INT max_len=-1;

				//usually n==num_vec, but it might not in race conditions 
				//(file perms modified, file erased)
				strings=new T_STRING<ST>[n];

				for (int i=0; i<n; i++)
				{
					char* fname=io.concat_filename(namelist[i]->d_name);

					struct stat s;
					off_t filesize=0;

					if (!stat(fname, &s) && s.st_size>0)
					{
						filesize=s.st_size/sizeof(ST);

						FILE* f=fopen(fname, "ro");
						if (f)
						{
							ST* str=new ST[filesize];
							SG_DEBUG("%s:%ld\n", fname, (long int) filesize);
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
		bool set_features(T_STRING<ST>* p_features, INT p_num_vectors, INT p_max_string_length)
		{
			if (p_features)
			{
				CAlphabet* alpha=new CAlphabet(alphabet);

				//compute histogram for char/byte
				for (INT i=0; i<p_num_vectors; i++)
					alpha->add_string_to_histogram( p_features[i].string, p_features[i].length);

				SG_INFO("max_value_in_histogram:%d\n", alpha->get_max_value_in_histogram());
				SG_INFO("num_symbols_in_histogram:%d\n", alpha->get_num_symbols_in_histogram());

				if (alpha->check_alphabet_size() && alpha->check_alphabet())
				{
					cleanup();

#ifdef HAVE_SWIG
					SG_UNREF(alphabet);
#else
					delete alphabet;
#endif
					alphabet=alpha;
					SG_REF(alphabet);

					this->features=p_features;
					this->num_vectors=p_num_vectors;
					this->max_string_length=p_max_string_length;

					return true;
				}
				else
					delete alpha;
			}

			return false;
		}

		/** get_features 
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual T_STRING<ST>* get_features(INT& num_str, INT& max_str_len)
		{
			num_str=num_vectors;
			max_str_len=max_string_length;
			return features;
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

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual INT get_size() { return sizeof(ST); }

		/** apply preprocessor
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preproc(bool force_preprocessing=false)
		{
			SG_DEBUG( "force: %d\n", force_preprocessing);

			for (INT i=0; i<get_num_preproc(); i++)
			{ 
				if ( (!is_preprocessed(i) || force_preprocessing) )
				{
					set_preprocessed(i);

					SG_INFO( "preprocessing using preproc %s\n", get_preproc(i)->get_name());

					if (!((CStringPreProc<ST>*) get_preproc(i))->apply_to_string_features(this))
						return false;
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
		INT obtain_by_sliding_window(INT window_size, INT step_size, INT skip=0)
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
			INT offs=0;
			for (INT i=0; i<num_vectors; i++)
			{
				f[i].string=&features[0].string[offs+skip];
				f[i].length=window_size-skip;
				offs+=step_size;
			}
			single_string=features[0].string;
			delete[] features;
			features=f;
			selected_vector=0;
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
		INT obtain_by_position_list(INT window_size, CDynamicArray<INT>* positions, INT skip=0)
		{
			ASSERT(positions);
			ASSERT(window_size>0);
			ASSERT(num_vectors==1 || single_string);
			ASSERT(max_string_length>=window_size ||
					(single_string && length_of_single_string>=window_size));

			num_vectors= positions->get_num_elements();
			ASSERT(num_vectors>0);

			INT len;

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
			for (INT i=0; i<num_vectors; i++)
			{
				INT p=positions->get_element(i);

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
			selected_vector=0;
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
		inline bool obtain_from_char(CStringFeatures<char>* sf, INT start, INT p_order, INT gap, bool rev)
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
			bool obtain_from_char_features(CStringFeatures<CT>* sf, INT start, INT p_order, INT gap, bool rev)
			{
				ASSERT(sf);
				this->order=p_order;
				cleanup();
				delete[] symbol_mask_table;
				symbol_mask_table=new ST[256];

				num_vectors=sf->get_num_vectors();
				ASSERT(num_vectors>0);
				max_string_length=sf->get_max_vector_length()-start;
				features=new T_STRING<ST>[num_vectors];
				CAlphabet* alpha=sf->get_alphabet();
				ASSERT(alpha->get_num_symbols_in_histogram() > 0);

				SG_DEBUG( "%1.0llf symbols in StringFeatures<*>\n", sf->get_num_symbols());

				for (INT i=0; i<num_vectors; i++)
				{
					INT len=-1;
					CT* c=sf->get_feature_vector(i, len);

					features[i].string=new ST[len];
					features[i].length=len;

					ST* str=features[i].string;
					for (INT j=0; j<len; j++)
						str[j]=(ST) alpha->remap_to_bin(c[j]);

				}

				original_num_symbols=alpha->get_num_symbols();
				INT max_val=alpha->get_num_bits();

				if (p_order>1)
					num_symbols=CMath::powl((long double) 2, (long double) max_val*p_order);
				else
					num_symbols=original_num_symbols;
				SG_INFO( "max_val (bit): %d order: %d -> results in num_symbols: %.0Lf\n", max_val, p_order, num_symbols);

				if ( ((long double) num_symbols) > CMath::powl(((long double) 2),((long double) sizeof(ST)*8)) )
				{
					SG_ERROR( "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
					return false;
				}

				SG_DEBUG( "translate: start=%i order=%i gap=%i(size:%i)\n", start, p_order, gap, sizeof(ST)) ;
				for (INT line=0; line<num_vectors; line++)
				{
					INT len=0;
					ST* fv=get_feature_vector(line, len);

					if (rev)
						translate_from_single_order_reversed(fv, len, start+gap, p_order+gap, max_val, gap);
					else
						translate_from_single_order(fv, len, start+gap, p_order+gap, max_val, gap);
					//translate_from_single_order(fv, len, start, p_order, max_val);
					//translate_from_single_order_reversed(fv, len, start, p_order, max_val);

					/* fix the length of the string -- hacky */
					features[line].length-=start+gap ;
					if (features[line].length<0)
						features[line].length=0 ;
				}         

				ULONG mask=0;
				for (INT i=0; i< (LONG) max_val; i++)
					mask=(mask<<1) | 1;

				for (INT i=0; i<256; i++)
				{
					uint8_t bits=(uint8_t) i;
					symbol_mask_table[i]=0;

					for (INT j=0; j<8; j++)
					{
						if (bits & 1)
							symbol_mask_table[i]|=mask<<(max_val*j);

						bits>>=1;
					}
				}

				return true;
			}

		/** check if length of each vector in this feature object equals the
		 * given length.
		 *
		 * @param len vector length to check against
		 * @return if length of each vector in this feature object equals the
		 * given length.
		 */
		bool have_same_length(INT len)
		{
			if (len!=get_max_vector_length())
				return false;

			for (INT i=0; i<num_vectors; i++)
			{
				if (get_vector_length(i)!=len)
					return false;
			}

			return true;
		}

	protected:
		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 */
		void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val)
		{
			INT i,j;
			ST value=0;

			for (i=sequence_length-1; i>= p_order-1; i--) //convert interval of size T
			{
				value=0;
				for (j=i; j>=i-p_order+1; j--)
					value= (value >> max_val) | (obs[j] << (max_val * (p_order-1)));

				obs[i]= (ST) value;
			}

			for (i=p_order-2;i>=0;i--)
			{
				if (i>=sequence_length)
					continue;

				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					value= (value >> max_val);
					if (j>=0 && j<sequence_length)
						value|=obs[j] << (max_val * (p_order-1));
				}
				obs[i]=value;
			}

			for (i=start; i<sequence_length; i++)
				obs[i-start]=obs[i];
		}

		/** translate from single order reversed
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 */
		void translate_from_single_order_reversed(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val)
		{
			INT i,j;
			ST value=0;

			for (i=sequence_length-1; i>= p_order-1; i--) //convert interval of size T
			{
				value=0;
				for (j=i; j>=i-p_order+1; j--)
					value= (value << max_val) | obs[j];

				obs[i]= (ST) value;
			}

			for (i=p_order-2;i>=0;i--)
			{
				if (i>=sequence_length)
					continue;

				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					value= (value << max_val);
					if (j>=0 && j<sequence_length)
						value|=obs[j];
				}
				obs[i]=value;
			}

			for (i=start; i<sequence_length; i++)
				obs[i-start]=obs[i];
		}

		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 * @param gap gap
		 */
		void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val, INT gap)
		{
			ASSERT(gap>=0);

			const INT start_gap=(p_order-gap)/2;
			const INT end_gap=start_gap+gap;

			INT i,j;
			ST value=0;

			// almost all positions
			for (i=sequence_length-1; i>=p_order-1; i--) //convert interval of size T
			{
				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					if (i-j<start_gap)
					{
						value= (value >> max_val) | (obs[j] << (max_val * (p_order-1-gap)));
					}
					else if (i-j>=end_gap)
					{
						value= (value >> max_val) | (obs[j] << (max_val * (p_order-1-gap)));
					}
				}
				obs[i]= (ST) value;
			}

			// the remaining `order` positions
			for (i=p_order-2;i>=0;i--)
			{
				if (i>=sequence_length)
					continue;

				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					if (i-j<start_gap)
					{
						value= (value >> max_val);
						if (j>=0 && j<sequence_length)
							value|=obs[j] << (max_val * (p_order-1-gap));
					}
					else if (i-j>=end_gap)
					{
						value= (value >> max_val);
						if (j>=0 && j<sequence_length)
							value|=obs[j] << (max_val * (p_order-1-gap));
					}
				}
				obs[i]=value;
			}

			// shifting
			for (i=start; i<sequence_length; i++)
				obs[i-start]=obs[i];
		}

		/** translate from single order reversed
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 * @param gap gap
		 */
		void translate_from_single_order_reversed(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val, INT gap)
		{
			ASSERT(gap>=0);

			const INT start_gap=(p_order-gap)/2;
			const INT end_gap=start_gap+gap;

			INT i,j;
			ST value=0;

			// almost all positions
			for (i=sequence_length-1; i>=p_order-1; i--) //convert interval of size T
			{
				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					if (i-j<start_gap)
						value= (value << max_val) | obs[j];
					else if (i-j>=end_gap)
						value= (value << max_val) | obs[j];
				}
				obs[i]= (ST) value;
			}

			// the remaining `order` positions
			for (i=p_order-2;i>=0;i--)
			{
				if (i>=sequence_length)
					continue;

				value=0;
				for (j=i; j>=i-p_order+1; j--)
				{
					if (i-j<start_gap)
					{
						value= value << max_val;
						if (j>=0 && j<sequence_length)
							value|=obs[j];
					}
					else if (i-j>=end_gap)
					{
						value= value << max_val;
						if (j>=0 && j<sequence_length)
							value|=obs[j];
					}			
				}
				obs[i]=value;
			}

			// shifting
			for (i=start; i<sequence_length; i++)
				obs[i-start]=obs[i];
		}

	protected:

		/// alphabet
		CAlphabet* alphabet;

		/// number of string vectors
		INT num_vectors;

		/// this contains the array of features.
		T_STRING<ST>* features;

		/// true when single string / created by sliding window
		ST* single_string;

		/// length of prior single string
		INT length_of_single_string;

		/// length of longest string
		INT max_string_length;

		/// number of used symbols
		LONGREAL num_symbols;

		/// original number of used symbols (before higher order mapping)
		LONGREAL original_num_symbols;

		/// order used in higher order mapping
		INT order;

		/// vector to be obtained via get_string
		INT selected_vector;

		/// order used in higher order mapping
		ST* symbol_mask_table;
};

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
template<> inline EFeatureType CStringFeatures<SHORT>::get_feature_type()
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
template<> inline EFeatureType CStringFeatures<INT>::get_feature_type()
{
	return F_INT;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CStringFeatures<UINT>::get_feature_type()
{
	return F_UINT;
}

/** get feature type the LONG feature can deal with
 *
 * @return feature type LONG
 */
template<> inline EFeatureType CStringFeatures<LONG>::get_feature_type()
{
	return F_LONG;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CStringFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CStringFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}
#endif
