/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include "preproc/PreProc.h"
#include "preproc/StringPreProc.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Mathmatics.h"

#include <math.h>


template <class T> struct T_STRING
{
	T* string;
	INT length;
};

// StringFeatures do not yet support PREPROCS
template <class ST> class CStringFeatures: public CFeatures
{
//	class CStringPreProc<ST> ;
	
	public:
	CStringFeatures(INT num_sym=-1) : CFeatures(0), num_vectors(0), features(NULL), max_string_length(0), num_symbols(num_sym), original_num_symbols(num_sym), order(0), symbol_mask_table(NULL)
	{
	}

	CStringFeatures(const CStringFeatures & orig) : CFeatures(orig), num_vectors(orig.num_vectors), max_string_length(orig.max_string_length), num_symbols(orig.num_symbols), original_num_symbols(orig.original_num_symbols), order(orig.order)
	{
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

	CStringFeatures(char* fname, E_ALPHABET alpha=NONE) : CFeatures(fname), num_vectors(0), features(NULL), max_string_length(0), num_symbols(0), original_num_symbols(0), order(0), symbol_mask_table(NULL)
	{
		alphabet_type=alpha;
		num_symbols=4; //FIXME
		load(fname);
	}

	virtual ~CStringFeatures()
	{
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

	virtual EFeatureClass get_feature_class() { return C_STRING ; } ;
	virtual EFeatureType get_feature_type();

	inline E_ALPHABET get_alphabet()
	{
		return alphabet_type;
	}

	virtual CFeatures* duplicate() const 
	{
		return new CStringFeatures<ST>(*this);
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

	inline INT get_num_symbols() { return num_symbols; }
	inline INT get_max_num_symbols() { return (1<<(sizeof(ST)*8)); }
	
	// these functions are necessary to find out about a former conversion process
	
	// number of symbols before higher order mapping
	inline INT get_original_num_symbols() { return original_num_symbols; }

	// order used when higher order mapping was done
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

				if (index+columns>=length && p[columns]!='\n')
					CIO::message(M_ERROR, "error in \"%s\":%d\n", fname, lines);

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
			CIO::message(M_ERROR, "reading file failed\n");

		return false;
	}

	void set_features(T_STRING<ST>* features, INT num_vectors, INT max_string_length, INT num_symbols, E_ALPHABET alphabet)
	{
		cleanup();
		this->features=features;
		this->num_vectors=num_vectors;
		this->max_string_length=max_string_length;
		this->num_symbols=num_symbols;
		this->alphabet_type=alphabet;
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
				
//					if (!((CStringPreProc<ST>*) get_preproc(i))->apply_to_feature_strings(this))
				if (!ok)
					return false;
			}
			return true;
		}

	bool obtain_from_char_features(CStringFeatures<CHAR>* sf, INT start, INT order)
	{
		INT i=0;
		ASSERT(sf);
		E_ALPHABET alphabet=sf->get_alphabet();
		this->order=order;
		cleanup();
		CCharFeatures cf(alphabet, 0);
		delete[] symbol_mask_table;
		symbol_mask_table=new ST[256];
		num_vectors=sf->get_num_vectors();
		max_string_length=sf->get_max_vector_length()-start;
		features=new T_STRING<ST>[num_vectors];
		ASSERT(features);

		CIO::message(M_DEBUG, "%d symbols in StringFeatures<CHAR>\n", sf->get_num_symbols());
		ST max_val=0;
		for (i=0; i<num_vectors; i++)
		{
			INT len=sf->get_vector_length(i);

			features[i].string=new ST[len];
			features[i].length=len;
			ASSERT(features[i].string);

			ST* str=features[i].string;

			for (INT j=0; j<len; j++)
			{
				str[j]=(ST) cf.remap(sf->get_feature(i,j));
				max_val=CMath::max((ST) str[j], (ST) max_val);
			}
		}

		original_num_symbols=max_val+1;

		//number of bits the maximum value in feature matrix requires to get stored
		max_val= (int) ceil(log((double) max_val+1)/log((double) 2));
		num_symbols=1<<(max_val*order);

		CIO::message(M_DEBUG, "max_val (bit): %d order: %d -> results in num_symbols: %d\n", max_val, order, num_symbols);

		if ( ((long double) num_symbols) > pow(((long double) 2),((long double) sizeof(ST)*8)) )
		{
			CIO::message(M_DEBUG, "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
			return false;
		}

		CIO::message(M_DEBUG, "translate: start=%i order=%i (size:%i)\n", start, order, sizeof(ST)) ;
		for (INT line=0; line<num_vectors; line++)
		{
			INT len=0;
			ST* fv=get_feature_vector(line, len);
			translate_from_single_order(fv, len, start, order, max_val);

			/* fix the length of the string -- hacky */
			features[line].length-=start ;
			if (features[line].length<0)
				features[line].length=0 ;
		}

		for (i=0; i<256; i++)
			symbol_mask_table[i]=0;

		ST mask=0;
		for (i=0; i<(INT) max_val; i++)
			mask=(mask<<1) | 1;

		for (i=0; i<256; i++)
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

	bool obtain_from_char_features(CStringFeatures<CHAR>* sf, E_ALPHABET alphabet, INT start, INT order, INT gap)
	{
		ASSERT(sf);
		this->order=order;
		cleanup();
		CCharFeatures cf(alphabet, 0);
		delete[] symbol_mask_table;
		symbol_mask_table=new ST[256];

		num_vectors=sf->get_num_vectors();
		max_string_length=sf->get_max_vector_length()-start;
		features=new T_STRING<ST>[num_vectors];
		ASSERT(features);

		CIO::message(M_DEBUG, "%d symbols in StringFeatures<CHAR>\n", sf->get_num_symbols());

		ST max_val=0;
		for (INT i=0; i<num_vectors; i++)
		{
			INT len=sf->get_vector_length(i);

			features[i].string=new ST[len];
			features[i].length=len;
			ASSERT(features[i].string);

			ST* str=features[i].string;

			for (INT j=0; j<len; j++)
			{
				str[j]=(ST) cf.remap(sf->get_feature(i,j));
				max_val=CMath::max((ST) str[j], (ST) max_val);
			}
		}

		original_num_symbols=max_val+1;
		
		INT* hist = new int[max_val+1] ;
		for (INT i=0; i<= (LONG) max_val; i++)
		  hist[i]=0 ;

		for (INT i=0; i<num_vectors; i++)
		{
			INT len=features[i].length;
			ST* str=features[i].string;
			
			for (INT j=0; j<len; j++)
				hist[str[j]]++;
		}
		for (INT i=0; i<=(LONG) max_val; i++)
		  if (hist[i]>0)
			CIO::message(M_DEBUG, "symbol: %i  number of occurence: %i\n", i, hist[i]) ;

		delete[] hist;

		//number of bits the maximum value in feature matrix requires to get stored
		max_val= (int) ceil(log((double) max_val+1)/log((double) 2));
		if (order>1)
			num_symbols=1<<(max_val*order);
		else
			num_symbols=original_num_symbols;

		CIO::message(M_INFO, "max_val (bit): %d order: %d -> results in num_symbols: %d\n", max_val, order, num_symbols);

		if ( ((long double) num_symbols) > pow(((long double) 2),((long double) sizeof(ST)*8)) )
		{
			CIO::message(M_ERROR, "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
			return false;
		}

		CIO::message(M_DEBUG, "translate: start=%i order=%i gap=%i(size:%i)\n", start, order, gap, sizeof(ST)) ;
		for (INT line=0; line<num_vectors; line++)
		{
			INT len=0;
			ST* fv=get_feature_vector(line, len);
			translate_from_single_order(fv, len, start+gap, order+gap, max_val, gap);

			/* fix the length of the string -- hacky */
			features[line].length-=start+gap ;
			if (features[line].length<0)
				features[line].length=0 ;
		}
		
		for (INT i=0; i<256; i++)
			symbol_mask_table[i]=0;

		WORD mask=0;
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

	void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT order, INT max_val)
	{
		INT i,j;
		ST value=0;

		for (i=sequence_length-1; i>= ((int) order)-1; i--)	//convert interval of size T
		{
			value=0;
			for (j=i; j>=i-((int) order)+1; j--)
				value= (value >> max_val) | (obs[j] << (max_val * (order-1)));

			obs[i]= (ST) value;
		}

		for (i=order-2;i>=0;i--)
		{
			value=0;
			for (j=i; j>=i-order+1; j--)
			{
				value= (value >> max_val);
				if (j>=0)
					value|=obs[j] << (max_val * (order-1));
			}
			obs[i]=value;
		}

		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
	}

	void translate_from_single_order(ST* obs, INT sequence_length, INT start, INT order, INT max_val, INT gap)
	{
		ASSERT(gap>=0) ;
		
		const INT start_gap = (order - gap)/2 ;
		const INT end_gap = start_gap + gap ;
		
		INT i,j;
		ST value=0;

		// almost all positions
		for (i=sequence_length-1; i>= ((int) order)-1; i--)	//convert interval of size T
		{
			value=0;
			for (j=i; j>=i-((int) order)+1; j--)
			{
				if (i-j<start_gap)
				{
					value= (value >> max_val) | (obs[j] << (max_val * (order-1-gap)));
				}
				else if (i-j>=end_gap)
				{
					value= (value >> max_val) | (obs[j] << (max_val * (order-1-gap)));
				}
			}
			obs[i]= (ST) value;
		}

		// the remaining `order` positions
		for (i=order-2;i>=0;i--)
		{
			value=0;
			for (j=i; j>=i-order+1; j--)
			{
				if (i-j<start_gap)
				{
					value= (value >> max_val);
					if (j>=0)
						value|=obs[j] << (max_val * (order-1-gap));
				}
				else if (i-j>=end_gap)
				{
					value= (value >> max_val);
					if (j>=0)
						value|=obs[j] << (max_val * (order-1-gap));
				}			
			}
			obs[i]=value;
		}

		// shifting
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
	}

	protected:
	/// number of string vectors
	INT num_vectors;

	/// this contains the array of features.
	T_STRING<ST>* features;

	/// length of longest string
	INT max_string_length;

	/// number of used symbols
	INT num_symbols;

	/// original number of used symbols (before higher order mapping)
	INT original_num_symbols;

	/// alphabet
	E_ALPHABET alphabet_type;

	/// order used in higher order mapping
	INT order;

	/// order used in higher order mapping
	ST* symbol_mask_table;
};

template<> inline EFeatureType CStringFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CStringFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

template<> inline EFeatureType CStringFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CStringFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}
#endif
