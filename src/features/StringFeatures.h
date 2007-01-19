/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
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
#include "lib/File.h"
#include "lib/Mathematics.h"

#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

template <class ST> class CStringPreProc;

template <class T> struct T_STRING
{
	T* string;
	INT length;
};

// StringFeatures do not yet support PREPROCS
template <class ST> class CStringFeatures: public CFeatures
{
	public:
	CStringFeatures(CAlphabet* alpha) : CFeatures(0), num_vectors(0), features(NULL), max_string_length(0), order(0), symbol_mask_table(NULL)
	{
		alphabet=new CAlphabet(alpha);
		ASSERT(alpha);
		num_symbols=alphabet->get_num_symbols();
		original_num_symbols=num_symbols;
	}

	CStringFeatures(const CStringFeatures & orig) : CFeatures(orig), num_vectors(orig.num_vectors), max_string_length(orig.max_string_length), num_symbols(orig.num_symbols), original_num_symbols(orig.original_num_symbols), order(orig.order)
	{
		alphabet=new CAlphabet(orig.alphabet);

		if (orig.features)
		{
			features=new T_STRING<ST>[orig.num_vectors];
			ASSERT(features);

			for (INT i=0; i<num_vectors; i++)
			{
				features[i].string=new ST[orig.features[i].length];
				ASSERT(features[i].string!=NULL);
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

	CStringFeatures(char* fname, E_ALPHABET alpha=DNA) : CFeatures(fname), num_vectors(0), features(NULL), max_string_length(0), order(0), symbol_mask_table(NULL)
	{
		alphabet=new CAlphabet(alpha);
		num_symbols=alphabet->get_num_symbols();
		original_num_symbols=num_symbols;
		load(fname);
	}

	virtual ~CStringFeatures()
	{
		delete alphabet;
		alphabet=NULL;
		cleanup();
	}

	void cleanup()
	{
		for (int i=0; i<num_vectors; i++)
		{
			delete[] features[i].string;
			features[i].length=0;
		}
		delete[] features;

		delete[] symbol_mask_table;
	}

	inline virtual EFeatureClass get_feature_class() { return C_STRING ; } ;
	inline virtual EFeatureType get_feature_type();

	inline CAlphabet* get_alphabet()
	{
		return alphabet;
	}

	virtual CFeatures* duplicate() const 
	{
		return new CStringFeatures<ST>(*this);
	}

	void select_feature_vector(INT num)
	{
		ASSERT(features!=NULL);
		ASSERT(num<num_vectors);

		selected_vector=num;
	}
	
	/** get feature vector for selected example
	  */
	void get_string(ST** dst, INT* len)
	{
		ASSERT(features!=NULL);
		ASSERT(selected_vector<num_vectors);

		*len=features[selected_vector].length;
		*dst=new ST[*len];
		memcpy(*dst, features[selected_vector].string, *len * sizeof(ST));
	}

	/** get feature vector for sample num
	  @param num index of feature vector
	  @param len length is returned by reference
	  */
	virtual ST* get_feature_vector(INT num, INT& len)
	{
		ASSERT(features!=NULL);
		ASSERT(num<num_vectors);

		len=features[num].length;
		return features[num].string;
	}

	/** get feature vector for sample num
	  @param num index of feature vector
	  @param len length is returned by reference
	  */
	virtual void set_feature_vector(INT num, ST* string, INT len)
	{
		ASSERT(features!=NULL);
		ASSERT(num<num_vectors);

		features[num].length=len ;
		features[num].string=string ;
	}

	virtual ST inline get_feature(INT vec_num, INT feat_num)
	{
		ASSERT(features && vec_num<num_vectors);
		ASSERT(feat_num < features[vec_num].length);

		return features[vec_num].string[feat_num];
	}

	virtual inline INT get_vector_length(INT vec_num)
	{
		ASSERT(features && vec_num<num_vectors);
		return features[vec_num].length;
	}

	virtual inline INT get_max_vector_length()
	{
		return max_string_length;
	}

	virtual inline INT get_num_vectors() { return num_vectors; }

	inline LONGREAL get_num_symbols() { return num_symbols; }
	inline LONGREAL get_max_num_symbols() { return CMath::powl(2,sizeof(ST)*8); }
	
	// these functions are necessary to find out about a former conversion process
	
	// number of symbols before higher order mapping
	inline LONGREAL get_original_num_symbols() { return original_num_symbols; }

	// order used for higher order mapping
	inline INT get_order() { return order; }

	// a higher order mapped symbol will be shaped such that the symbols in
	// specified by bits in the mask will be returned.
	inline ST get_masked_symbols(ST symbol, BYTE mask)
	{
		ASSERT(symbol_mask_table);
		return symbol_mask_table[mask] & symbol;
	}

	virtual bool load(CHAR* fname)
	{
		CIO::message(M_INFO, "loading...\n");
		LONG length=0;
		max_string_length=0;

		CFile f(fname, 'r', F_CHAR);
		CHAR* feature_matrix=f.load_char_data(NULL, length);

		num_vectors=0;

		if (f.is_ok())
		{
			for (long i=0; i<length; i++)
			{
				if (feature_matrix[i]=='\n')
					num_vectors++;
			}

			CIO::message(M_INFO, "file contains %ld vectors\n", num_vectors);
			features= new T_STRING<ST>[num_vectors];

			long index=0;
			for (INT lines=0; lines<num_vectors; lines++)
			{
				CHAR* p=&feature_matrix[index];
				INT columns=0;

				for (columns=0; index+columns<length && p[columns]!='\n'; columns++);

				if (index+columns>=length && p[columns]!='\n') {
#ifdef HAVE_PYTHON
               throw FeatureException("error in \"%s\":%d\n", fname, lines);
#else
					CIO::message(M_ERROR, "error in \"%s\":%d\n", fname, lines);
#endif
            }

				features[lines].length=columns;
				features[lines].string=new ST[columns];
				ASSERT(features[lines].string);

				max_string_length=CMath::max(max_string_length,columns);

				for (INT i=0; i<columns; i++)
					features[lines].string[i]= ((ST) p[i]);

				index+= features[lines].length+1;
			}

			num_symbols=4; //FIXME
			return true;
		}
		else
#ifdef HAVE_PYTHON
         throw FeatureException("reading file failed\n");
#else
			CIO::message(M_ERROR, "reading file failed\n");
#endif

		return false;
	}

	bool load_from_directory(CHAR* dirname)
	{
		struct dirent **namelist;
		int n;

		CIO::set_dirname(dirname);

		n = scandir(dirname, &namelist, CIO::filter, alphasort);
		if (n <= 0)
		{
			CIO::message(M_ERROR, "error calling scandir\n");
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
				CHAR* fname=CIO::concat_filename(namelist[i]->d_name);

				struct stat s;
				off_t filesize=0;

				if (!stat(fname, &s) && s.st_size>0)
				{
					filesize=s.st_size/sizeof(ST);

					FILE* f=fopen(fname, "ro");
					if (f)
					{
						ST* str=new ST[filesize];
						ASSERT(str);
						CIO::message(M_DEBUG,"%s:%ld\n", fname, (long int) filesize);
						fread(str, sizeof(ST), filesize, f);
						strings[num].string=str;
						strings[num].length=filesize;
						max_len=CMath::max(max_len, strings[num].length);

						//compute histogram for char/byte
						alphabet->add_string_to_histogram((BYTE*) strings[num].string, (LONG) strings[num].length);
						
						num++;
						fclose(f);
					}
				}
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

	void set_features(T_STRING<ST>* p_features, INT p_num_vectors, INT p_max_string_length)
	{
		cleanup();
		this->features=p_features;
		this->num_vectors=p_num_vectors;
		this->max_string_length=p_max_string_length;
	}

	void copy_features(T_STRING<ST>* p_features, INT p_num_vectors, INT p_max_string_length)
	{
		cleanup();
		this->features=p_features;
		this->num_vectors=p_num_vectors;
		this->max_string_length=p_max_string_length;
	}

	virtual bool save(CHAR* dest)
	{
		return false;
	}

	virtual INT get_size() { return sizeof(ST); }

	/// preprocess the feature_matrix
	virtual bool preproc_feature_strings(bool force_preprocessing=false)
	{
		CIO::message(M_DEBUG, "force: %d\n", force_preprocessing);

		for (INT i=0; i<get_num_preproc(); i++)
		{ 
			CIO::message(M_INFO, "preprocessing using preproc %s\n", get_preproc(i)->get_name());
			bool ok=((CStringPreProc<ST>*) get_preproc(i))->apply_to_feature_strings(this) ;

			if (!ok)
				return false;
		}
		return true;
	}

	inline bool obtain_from_char(CStringFeatures<CHAR>* sf, INT start, INT p_order, INT gap)
	{
		return obtain_from_char_features(sf, start, p_order, gap);
	}

    template <class CT>
	bool obtain_from_char_features(CStringFeatures<CT>* sf, INT start, INT p_order, INT gap)
	{
		ASSERT(sf);
		this->order=order;
		cleanup();
		delete[] symbol_mask_table;
		symbol_mask_table=new ST[256];

		num_vectors=sf->get_num_vectors();
		ASSERT(num_vectors>0);
		max_string_length=sf->get_max_vector_length()-start;
		features=new T_STRING<ST>[num_vectors];
		ASSERT(features);

		CAlphabet* alpha=sf->get_alphabet();
		//alpha->clear_histogram();
		//for (INT i=0; i<num_vectors; i++)
		//{
		//	INT len=-1;
		//	CT* c=sf->get_feature_vector(i, len);
		//	alpha->add_string_to_histogram(c, len);
		//}

		ASSERT(alpha->get_num_symbols_in_histogram() > 0);

		CIO::message(M_DEBUG, "%1.0llf symbols in StringFeatures<*>\n", sf->get_num_symbols());

		for (INT i=0; i<num_vectors; i++)
		{
			INT len=-1;
			CT* c=sf->get_feature_vector(i, len);

			features[i].string=new ST[len];
			features[i].length=len;
			ASSERT(features[i].string);

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
		CIO::message(M_INFO, "max_val (bit): %d order: %d -> results in num_symbols: %.0Lf\n", max_val, p_order, num_symbols);

		if ( ((long double) num_symbols) > CMath::powl(((long double) 2),((long double) sizeof(ST)*8)) )
		{
#ifdef HAVE_PYTHON
         throw FeatureException("symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
#else
			CIO::message(M_ERROR, "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
#endif
			return false;
		}

		CIO::message(M_DEBUG, "translate: start=%i order=%i gap=%i(size:%i)\n", start, p_order, gap, sizeof(ST)) ;
		for (INT line=0; line<num_vectors; line++)
		{
			INT len=0;
			ST* fv=get_feature_vector(line, len);
			translate_from_single_order(fv, len, start+gap, p_order+gap, max_val, gap);

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
			BYTE bits=(BYTE) i;

			for (INT j=0; j<8; j++)
			{
				if (bits & 1)
					symbol_mask_table[i]|=mask<<(max_val*j);

				bits>>=1;
			}
		}

		return true;
	}

	protected:

	void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val)
	{
		INT i,j;
		ST value=0;

		for (i=sequence_length-1; i>= ((int) p_order)-1; i--)	//convert interval of size T
		{
			value=0;
			for (j=i; j>=i-((int) p_order)+1; j--)
				value= (value >> max_val) | (obs[j] << (max_val * (p_order-1)));

			obs[i]= (ST) value;
		}

		for (i=p_order-2;i>=0;i--)
		{
			value=0;
			for (j=i; j>=i-p_order+1; j--)
			{
				value= (value >> max_val);
				if (j>=0)
					value|=obs[j] << (max_val * (p_order-1));
			}
			obs[i]=value;
		}

		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
	}

	void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT p_order, INT max_val, INT gap)
	{
		ASSERT(gap>=0) ;
		
		const INT start_gap = (p_order - gap)/2 ;
		const INT end_gap = start_gap + gap ;
		
		INT i,j;
		ST value=0;

		// almost all positions
		for (i=sequence_length-1; i>= ((int) p_order)-1; i--)	//convert interval of size T
		{
			value=0;
			for (j=i; j>=i-((int) p_order)+1; j--)
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
			value=0;
			for (j=i; j>=i-p_order+1; j--)
			{
				if (i-j<start_gap)
				{
					value= (value >> max_val);
					if (j>=0)
						value|=obs[j] << (max_val * (p_order-1-gap));
				}
				else if (i-j>=end_gap)
				{
					value= (value >> max_val);
					if (j>=0)
						value|=obs[j] << (max_val * (p_order-1-gap));
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

template<> inline EFeatureType CStringFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CStringFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

template<> inline EFeatureType CStringFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringFeatures<INT>::get_feature_type()
{
	return F_INT;
}

template<> inline EFeatureType CStringFeatures<UINT>::get_feature_type()
{
	return F_UINT;
}

template<> inline EFeatureType CStringFeatures<LONG>::get_feature_type()
{
	return F_LONG;
}

template<> inline EFeatureType CStringFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}

template<> inline EFeatureType CStringFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}
#endif
