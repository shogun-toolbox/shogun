/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CALPHABET__H__
#define _CALPHABET__H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{
/// Alphabet of charfeatures/observations
enum EAlphabet
{
	/// DNA - letters A,C,G,T
	DNA=0,

	/// RAWDNA - letters 0,1,2,3
	RAWDNA=1,

	/// RNA - letters A,C,G,U
	RNA=2,

	/// PROTEIN - letters A-Z
	PROTEIN=3,

	// BINARY just 0 and 1
	BINARY=4,

	/// ALPHANUM - [0-9A-Z]
	ALPHANUM=5,

	/// CUBE - [1-6]
	CUBE=6,

	/// RAW BYTE - [0-255]
	RAWBYTE=7,

	/// IUPAC_NUCLEIC_ACID
	IUPAC_NUCLEIC_ACID=8,

	/// IUPAC_AMINO_ACID
	IUPAC_AMINO_ACID=9,

	/// NONE - type has no alphabet
	NONE=10,

	/// DIGIT - letters 0-9
	DIGIT=11,

	/// DIGIT2 - letters 0-2
	DIGIT2=12,

	/// RAWDIGIT - 0-9
	RAWDIGIT=13,

	/// RAWDIGIT2 - 0-2
	RAWDIGIT2=14,

	/// unknown alphabet
	UNKNOWN=15,

	/// SNP - letters A,C,G,T,0
	SNP=16,

	/// RAWSNP - letters 0,1,2,3,4
	RAWSNP=17
};


/** @brief The class Alphabet implements an alphabet and alphabet utility functions.
 *
 * These utility functions can be used to remap characters to more
 * (bit-)efficient representations, check if a string is valid, compute
 * histograms etc.
 *
 * Currently supported alphabets are DNA, RAWDNA, RNA, PROTEIN,
 * BINARY, ALPHANUM, CUBE, RAW, IUPAC_NUCLEIC_ACID and IUPAC_AMINO_ACID.
 *
 */
class CAlphabet : public CSGObject
{
	public:

		/** default constructor
		 *
		 */
		CAlphabet();

		/** constructor
		 *
		 * @param alpha alphabet to use
		 * @param len len
		 */
		CAlphabet(char* alpha, int32_t len);

		/** constructor
		 *
		 * @param alpha alphabet (type) to use
		 */
		CAlphabet(EAlphabet alpha);

		/** constructor
		 *
		 * @param alpha alphabet to use
		 */
		CAlphabet(CAlphabet* alpha);
		virtual ~CAlphabet();

		/** set alphabet and initialize mapping table (for remap)
		 *
		 * @param alpha new alphabet
		 */
		bool set_alphabet(EAlphabet alpha);

		/** get alphabet
		 *
		 * @return alphabet
		 */
		inline EAlphabet get_alphabet() const
		{
			return alphabet;
		}

		/** get number of symbols in alphabet
		 *
		 * @return number of symbols
		 */
		inline int32_t get_num_symbols() const
		{
			return num_symbols;
		}

		/** get number of bits necessary to store
		 * all symbols in alphabet
		 *
		 * @return number of necessary storage bits
		 */
		inline int32_t get_num_bits() const
		{
			return num_bits;
		}

		/** remap element e.g translate ACGT to 0123
		 *
		 * @param c element to remap
		 * @return remapped element
		 */
		inline uint8_t remap_to_bin(uint8_t c)
		{
			return maptable_to_bin[c];
		}

		/** remap element e.g translate 0123 to ACGT
		 *
		 * @param c element to remap
		 * @return remapped element
		 */
		inline uint8_t remap_to_char(uint8_t c)
		{
			return maptable_to_char[c];
		}

		/// clear histogram
		void clear_histogram();

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		template <class T>
		void add_string_to_histogram(T* p, int64_t len)
		{
			for (int64_t i=0; i<len; i++)
				add_byte_to_histogram((uint8_t) (p[i]));
		}

		/** add element to histogram
		 *
		 * @param p element
		 */
		inline void add_byte_to_histogram(uint8_t p)
		{
			histogram[p]++;
		}

		/// print histogram
		void print_histogram();

		/** get histogram
		 *
		 * @return histogram
		 */
		SGVector<int64_t> get_histogram();

		/** check whether symbols in histogram are valid in alphabet
		 * e.g. for DNA if only letters ACGT appear
		 *
		 * @param print_error if errors shall be printed
		 * @return if symbols in histogram are valid in alphabet
		 */
		bool check_alphabet(bool print_error=true);

		/** check whether symbols are valid in alphabet
		 * e.g. for DNA if symbol is one of the A,C,G or T
		 *
		 * @param c symbol
		 * @return if symbol is a valid character in alphabet
		 */
		inline bool is_valid(uint8_t c)
		{
			return valid_chars[c];
		}

		/** check whether symbols in histogram ALL fit in alphabet
		 *
		 * @param print_error if errors shall be printed
		 * @return if symbols in histogram ALL fit in alphabet
		 */
		bool check_alphabet_size(bool print_error=true);

		/** return number of symbols in histogram
		 *
		 * @return number of symbols in histogram
		 */
		int32_t get_num_symbols_in_histogram();

		/** return maximum value in histogram
		 *
		 * @return maximum value in histogram
		 */
		int32_t get_max_value_in_histogram();

		/** return number of bits required to store all symbols in
		 * histogram
		 *
		 * @return number of bits required to store all symbols in
		 *         histogram
		 */
		int32_t get_num_bits_in_histogram();

		/** return alphabet name
		 *
		 * @param alphabet alphabet type to get name from
		 */
		static const char* get_alphabet_name(EAlphabet alphabet);


		/** @return object name */
		virtual const char* get_name() const { return "Alphabet"; }

		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 */
		template <class ST>
		static void translate_from_single_order(ST* obs, int32_t sequence_length, int32_t start, int32_t p_order, int32_t max_val);

		/** translate from single order reversed
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 */
		template <class ST>
		static void translate_from_single_order_reversed(ST* obs, int32_t sequence_length, int32_t start, int32_t p_order, int32_t max_val);

		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 * @param gap gap
		 */
		template <class ST>
		static void translate_from_single_order(ST* obs, int32_t sequence_length, int32_t start, int32_t p_order, int32_t max_val, int32_t gap);

		/** translate from single order reversed
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param p_order order
		 * @param max_val maximum value
		 * @param gap gap
		 */
		template <class ST>
		static void translate_from_single_order_reversed(ST* obs, int32_t sequence_length, int32_t start, int32_t p_order, int32_t max_val, int32_t gap);

	private:
		/** Do basic initialisations like default settings
		 * and registering parameters */
		void init();

	protected:
		/** init map table */
		void init_map_table();

		/** copy histogram
		 *
		 * @param src alphabet to copy histogram from
		 */
		void copy_histogram(CAlphabet* src);

	public:
		/** B_A */
		static const uint8_t B_A;
		/** B_C */
		static const uint8_t B_C;
		/** B_G */
		static const uint8_t B_G;
		/** B_T */
		static const uint8_t B_T;
		/** B_0 */
		static const uint8_t B_0;
		/** MAPTABLE UNDEF */
		static const uint8_t MAPTABLE_UNDEF;
		/** alphabet names */
		static const char* alphabet_names[18];

	protected:
		/** Can (optionally) be overridden to post-initialize some
		 *  member variables which are not PARAMETER::ADD'ed.  Make
		 *  sure that at first the overridden method
		 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_post() throw (ShogunException);

	protected:
		/** alphabet */
		EAlphabet alphabet;
		/** number of symbols */
		int32_t num_symbols;
		/** number of bits */
		int32_t num_bits;
		/** valid chars */
		bool valid_chars[1 << (sizeof(uint8_t)*8)];
		/** maptable to bin */
		uint8_t maptable_to_bin[1 << (sizeof(uint8_t)*8)];
		/** maptable to char */
		uint8_t maptable_to_char[1 << (sizeof(uint8_t)*8)];
		/** histogram */
		int64_t histogram[1 << (sizeof(uint8_t)*8)];
};
}
#endif
