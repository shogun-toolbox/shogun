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
const BYTE CAlphabet::B_star=4;
const BYTE CAlphabet::B_N=5;
const BYTE CAlphabet::B_n=6;
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
		alpha = RAW;
	
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
		case RAW:
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
		maptable[i] = MAPTABLE_UNDEF ;

	switch (alphabet)
	{
		case CUBE:
			maptable[(BYTE) '1']=0;
			maptable[(BYTE) '2']=1;
			maptable[(BYTE) '3']=2;
			maptable[(BYTE) '4']=3;	
			maptable[(BYTE) '5']=4;	
			maptable[(BYTE) '6']=5;	//Translation '123456' -> 012345

			maptable[(BYTE) 0]='1';
			maptable[(BYTE) 1]='2';
			maptable[(BYTE) 2]='3';
			maptable[(BYTE) 3]='4';
			maptable[(BYTE) 4]='5';
			maptable[(BYTE) 5]='6';	//Translation 012345->'123456'

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
					maptable[i]='a'+i+skip ;
					maptable['a'+i+skip]=i ;
					//printf("maptable[%c]=%i\n", 'a'+i+skip, i) ;
				} ;                   //Translation 012345->acde...xy -- the protein code
			} ;
			break;
		case ALPHANUM:
			{
				for (i=0; i<26; i++)
				{
					maptable[i]='a'+i ;
					maptable['a'+i]=i ;
				} ;
				for (i=0; i<10; i++)
				{
					maptable[26+i]='0'+i ;
					maptable['0'+i]=26+i ;
				} ;        //Translation 012345->acde...xy0123456789
			} ;
			break;
		case RAW:
			{
				//identity
				for (i=0; i<256; i++)
					maptable[i]=i;
			}
			break;
		case DNA:
			maptable[(BYTE) 'A']=B_A;
			maptable[(BYTE) 'C']=B_C;
			maptable[(BYTE) 'G']=B_G;
			maptable[(BYTE) 'T']=B_T;	
			maptable[(BYTE) '*']=B_star;	
			maptable[(BYTE) 'N']=B_N;	
			maptable[(BYTE) 'n']=B_n;	//Translation ACGTNn -> 012345

			maptable[B_A]='A';
			maptable[B_C]='C';
			maptable[B_G]='G';
			maptable[B_T]='T';
			maptable[B_star]='*';
			maptable[B_N]='N';
			maptable[B_n]='n';	//Translation 012345->ACGTNn
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
	for (INT i=(INT) (1 <<(sizeof(BYTE)*8));i>0; --i)
	{
		if (histogram[i]>0)
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
		if (histogram[i]>0)
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
			CIO::message(M_MESSAGEONLY, "hist[%d]=%d\n", i, histogram[i]);
	}
}

bool CAlphabet::check_alphabet_size()
{
	if (get_num_bits_in_histogram() > get_num_bits())
	{
		CIO::message(M_WARN, "ALPHABET too small to contain all symbols in histogram\n");
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
		case RAW:
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
