/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>
#include <math.h>

#include "features/Alphabet.h"
#include "lib/io.h"

//define numbers for the bases 
const BYTE CAlphabet::B_A=0;
const BYTE CAlphabet::B_C=1;
const BYTE CAlphabet::B_G=2;
const BYTE CAlphabet::B_T=3;
const BYTE CAlphabet::MAPTABLE_UNDEF=0xff;
const CHAR* CAlphabet::alphabet_names[]={"DNA", "PROTEIN", "ALPHANUM", "CUBE", "RAW", "NONE", "UNKNOWN"};

CAlphabet::CAlphabet(CHAR* al, INT len)
{
	E_ALPHABET alpha=NONE;

	if (len>=(INT) strlen("DNA") && !strncmp(al, "DNA", strlen("DNA")))
		alpha = DNA;
	else if (len>=(INT) strlen("PROTEIN") && !strncmp(al, "PROTEIN", strlen("PROTEIN")))
		alpha = PROTEIN;
	else if (len>=(INT) strlen("ALPHANUM") && !strncmp(al, "ALPHANUM", strlen("ALPHANUM")))
		alpha = ALPHANUM;
	else if (len>=(INT) strlen("CUBE") && !strncmp(al, "CUBE", strlen("CUBE")))
		alpha = CUBE;
	else if (len>=(INT) strlen("BYTE") && !strncmp(al, "BYTE", strlen("BYTE")) || 
			(len>=(INT) strlen("RAW") && !strncmp(al, "RAW", strlen("RAW"))))
		alpha = RAWBYTE;
	else
		CIO::message(M_ERROR, "unknown alphabet %s\n", al);
	
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(E_ALPHABET alpha)
{
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(CAlphabet* a)
{
	set_alphabet(a->get_alphabet());
}

CAlphabet::~CAlphabet()
{
}

bool CAlphabet::set_alphabet(E_ALPHABET alpha)
{
	bool result=true;
	alphabet=alpha;

	switch (alphabet)
	{
		case DNA:
			num_symbols = 4;
			break;
		case PROTEIN:
			num_symbols = 26;
			break;
		case ALPHANUM:
			num_symbols = 36;
			break;
		case CUBE:
			num_symbols = 6;
			break;
		case RAWBYTE:
			num_symbols = 256;
			break;
		case NONE:
			num_symbols = 0;
			break;
		default:
			num_symbols = 0;
			result=false;
			break;
	}

	num_bits=(INT) ceil(log((double) num_symbols)/log((double) 2));
	init_map_table();

	CIO::message(M_DEBUG, "initialised alphabet %s\n", get_alphabet_name(alphabet));

	return result;
}

void CAlphabet::init_map_table()
{
	INT i;
	for (i=0; i<(1<<(8*sizeof(BYTE))); i++)
	{
		maptable_to_bin[i] = MAPTABLE_UNDEF;
		maptable_to_char[i] = MAPTABLE_UNDEF;
		valid_chars[i] = 0;
	}

	switch (alphabet)
	{
		case CUBE:
			valid_chars[(BYTE) '1']=1;
			valid_chars[(BYTE) '2']=1;
			valid_chars[(BYTE) '3']=1;
			valid_chars[(BYTE) '4']=1;	
			valid_chars[(BYTE) '5']=1;	
			valid_chars[(BYTE) '6']=1;	//Translation '123456' -> 012345

			maptable_to_bin[(BYTE) '1']=0;
			maptable_to_bin[(BYTE) '2']=1;
			maptable_to_bin[(BYTE) '3']=2;
			maptable_to_bin[(BYTE) '4']=3;	
			maptable_to_bin[(BYTE) '5']=4;	
			maptable_to_bin[(BYTE) '6']=5;	//Translation '123456' -> 012345

			maptable_to_char[(BYTE) 0]='1';
			maptable_to_char[(BYTE) 1]='2';
			maptable_to_char[(BYTE) 2]='3';
			maptable_to_char[(BYTE) 3]='4';
			maptable_to_char[(BYTE) 4]='5';
			maptable_to_char[(BYTE) 5]='6';	//Translation 012345->'123456'

			break;
		case PROTEIN:
			{
				INT skip=0 ;
				for (i=0; i<21; i++)
				{
					if (i==1) skip++ ;
					if (i==8) skip++ ;
					if (i==12) skip++ ;
					if (i==17) skip++ ;
					valid_chars['a'+i+skip]=1;
					maptable_to_bin['a'+i+skip]=i ;
					maptable_to_char[i]='a'+i+skip ;
				} ;                   //Translation 012345->acde...xy -- the protein code
			} ;
			break;
		case ALPHANUM:
			{
				for (i=0; i<26; i++)
				{
					valid_chars['a'+i]=1;
					maptable_to_bin['a'+i]=i ;
					maptable_to_char[i]='a'+i ;
				} ;
				for (i=0; i<10; i++)
				{
					valid_chars['0'+i]=1;
					maptable_to_bin['0'+i]=26+i ;
					maptable_to_char[26+i]='0'+i ;
				} ;        //Translation 012345->acde...xy0123456789
			} ;
			break;
		case RAWBYTE:
			{
				//identity
				for (i=0; i<256; i++)
				{
					valid_chars[i]=1;
					maptable_to_char[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;
		case DNA:
			valid_chars[(BYTE) 'A']=B_A;
			valid_chars[(BYTE) 'C']=B_C;
			valid_chars[(BYTE) 'G']=B_G;
			valid_chars[(BYTE) 'T']=B_T;	

			maptable_to_bin[(BYTE) 'A']=B_A;
			maptable_to_bin[(BYTE) 'C']=B_C;
			maptable_to_bin[(BYTE) 'G']=B_G;
			maptable_to_bin[(BYTE) 'T']=B_T;	

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='T';
			break;
		default:
			break; //leave uninitialised
	};
}

void CAlphabet::clear_histogram()
{
	memset(histogram, 0, sizeof(LONG) * (1 << (sizeof(BYTE)*8)));
}

void CAlphabet::add_string_to_histogram(BYTE* p, INT len)
{
	for (INT i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

void CAlphabet::add_string_to_histogram(CHAR* p, INT len)
{
	for (INT i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

INT CAlphabet::get_max_value_in_histogram()
{
	INT max_sym=-1;
	for (INT i=(INT) (1 <<(sizeof(BYTE)*8))-1;i>=0; i--)
	{
		if (histogram[i])
		{
			max_sym=i;
			break;
		}
	}

	return max_sym;
}

INT CAlphabet::get_num_symbols_in_histogram()
{
	INT num_sym=0;
	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
	{
		if (histogram[i])
			num_sym++;
	}

	return num_sym;
}

INT CAlphabet::get_num_bits_in_histogram()
{
	INT num_sym=get_num_symbols_in_histogram();
	if (num_sym>0)
		return (INT) ceil(log((double) num_sym)/log((double) 2));
	else
		return 0;
}

void CAlphabet::print_histogram()
{
	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
	{
		if (histogram[i])
			CIO::message(M_MESSAGEONLY, "hist[%d]=%ld\n", i, histogram[i]);
	}
}

bool CAlphabet::check_alphabet(bool print_error)
{
	bool result = true;

	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
	{
		if (histogram[i]>0 && valid_chars[i]==0)
		{
			result=false;
			break;
		}
	}

	if (!result && print_error)
	{
		print_histogram();
		CIO::message(M_ERROR, "ALPHABET does not contain all symbols in histogram\n");
	}

	return result;
}

bool CAlphabet::check_alphabet_size(bool print_error)
{
	if (get_num_bits_in_histogram() > get_num_bits())
	{
		if (print_error)
		{
			print_histogram();
			CIO::message(M_ERROR, "ALPHABET too small to contain all symbols in histogram\n");
		}
		return false;
	}
	else
		return true;

}

const CHAR* CAlphabet::get_alphabet_name(E_ALPHABET alphabet)
{
	
	INT idx;
	switch (alphabet)
	{
		case DNA:
			idx=0;
			break;
		case PROTEIN:
			idx=1;
			break;
		case ALPHANUM:
			idx=2;
			break;
		case CUBE:
			idx=3;
			break;
		case RAWBYTE:
			idx=4;
			break;
		case NONE:
			idx=5;
			break;
		default:
			idx=6;
			break;
	}
	return alphabet_names[idx];
}
