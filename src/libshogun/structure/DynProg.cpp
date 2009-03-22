/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 2008-2009 Jonas Behr
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "structure/DynProg.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "lib/config.h"
#include "features/StringFeatures.h"
#include "features/Alphabet.h"
#include "structure/Plif.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

template void CDynProg::best_path_trans<1,true,false>(
	const float64_t *seq, int32_t seq_len, const int32_t *pos,
	const int32_t *orf_info, CPlifBase **PLif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,
	int32_t genestr_num, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, bool use_orf);

template void CDynProg::best_path_trans<2,true,false>(
	const float64_t *seq, int32_t seq_len, const int32_t *pos,
	const int32_t *orf_info, CPlifBase **PLif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,
	int32_t genestr_num, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, bool use_orf);

template void CDynProg::best_path_trans<1,false,false>(
	const float64_t *seq, int32_t seq_len, const int32_t *pos,
	const int32_t *orf_info, CPlifBase **PLif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,
	int32_t genestr_num, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, bool use_orf);

template void CDynProg::best_path_trans<2,false,false>(
	const float64_t *seq, int32_t seq_len, const int32_t *pos,
	const int32_t *orf_info, CPlifBase **PLif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,
	int32_t genestr_num, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, bool use_orf);


//#define USE_TMP_ARRAYCLASS
//#define DYNPROG_DEBUG

//CArray2<int32_t> g_orf_info(1,1) ;

static int32_t word_degree_default[4]={3,4,5,6} ;
static int32_t cum_num_words_default[5]={0,64,320,1344,5440} ;
static int32_t num_words_default[4]=   {64,256,1024,4096} ;
static int32_t mod_words_default[32] = {1,1,1,1,1,1,1,1,
									1,1,1,1,1,1,1,1,
									0,0,0,0,0,0,0,0,
									0,0,0,0,0,0,0,0} ;  
static bool sign_words_default[16] = {true,true,true,true,true,true,true,true,
									  false,false,false,false,false,false,false,false} ; // whether to use counts or signum of counts
static int32_t string_words_default[16] = {0,0,0,0,0,0,0,0,
									   1,1,1,1,1,1,1,1} ; // which string should be used

CDynProg::CDynProg(int32_t p_num_svms /*= 8 */)
: CSGObject(), transition_matrix_a_id(1,1), transition_matrix_a(1,1),
	transition_matrix_a_deriv(1,1), initial_state_distribution_p(1),
	initial_state_distribution_p_deriv(1), end_state_distribution_q(1),
	end_state_distribution_q_deriv(1), dict_weights(1,1),
	dict_weights_array(dict_weights.get_array()),

	  // multi svm
	  num_degrees(4), 
	  num_svms(p_num_svms), 
	  num_strings(1),
	  word_degree(word_degree_default, num_degrees, true, true),
	  cum_num_words(cum_num_words_default, num_degrees+1, true, true),
	  cum_num_words_array(cum_num_words.get_array()),
	  num_words(num_words_default, num_degrees, true, true),
	  num_words_array(num_words.get_array()),
	  mod_words(mod_words_default, num_svms, 2, true, true),
	  mod_words_array(mod_words.get_array()),
	  sign_words(sign_words_default, num_svms, true, true),
	  sign_words_array(sign_words.get_array()),
	  string_words(string_words_default, num_svms, true, true),
	  string_words_array(string_words.get_array()),
//	  word_used(num_degrees, num_words[num_degrees-1], num_strings),
//	  word_used_array(word_used.get_array()),
//	  svm_values_unnormalized(num_degrees, num_svms),
	  svm_pos_start(num_degrees),
	  num_unique_words(num_degrees),
	  svm_arrays_clean(true),

	  // single svm
	  num_svms_single(1),
	  word_degree_single(1),
	  num_words_single(4), 
	  word_used_single(num_words_single),
	  svm_value_unnormalized_single(num_svms_single),
	  num_unique_words_single(0),

	  max_a_id(0), m_seq(1,1,1), m_pos(1), m_orf_info(1,2), 
          m_segment_sum_weights(1,1), m_plif_list(1), 
	  m_PEN(1,1), m_PEN_state_signals(1,1), 
	  m_genestr(1,1), m_dict_weights(1,1), m_segment_loss(1,1,2), 
          m_segment_ids(1),
          m_segment_mask(1),
	  m_scores(1), m_states(1,1), m_positions(1,1), m_genestr_stop(1),
	  m_lin_feat(1,1) //by Jonas
	  //m_num_lin_feat(num_svms)
{
	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_forward_id = NULL ;
	trans_list_len = 0 ;

	mem_initialized = true ;

	this->N=1;
	m_step=0;

	m_raw_intensities = NULL;
	m_probe_pos = NULL;
	m_num_probes_cum = new int32_t[100];
	m_num_probes_cum[0] = 0;
	//m_use_tiling=false;
	m_num_lin_feat_plifs_cum = new int32_t[100];
	m_num_lin_feat_plifs_cum[0] = num_svms;
	m_num_raw_data = 0;
	m_genestr_len = 0;
#ifdef ARRAY_STATISTICS
	word_degree.set_name("word_degree");
#endif
}

CDynProg::~CDynProg()
{
	if (trans_list_forward_cnt)
		delete[] trans_list_forward_cnt;
	if (trans_list_forward)
	{
		for (int32_t i=0; i<trans_list_len; i++)
		{
			if (trans_list_forward[i])
				delete[] trans_list_forward[i];
		}
		delete[] trans_list_forward;
	}
	if (trans_list_forward_val)
	{
		for (int32_t i=0; i<trans_list_len; i++)
		{
			if (trans_list_forward_val[i])
				delete[] trans_list_forward_val[i];
		}
		delete[] trans_list_forward_val;
	}
	if (trans_list_forward_id)
	{
		for (int32_t i=0; i<trans_list_len; i++)
		{
			if (trans_list_forward_id[i])
				delete[] trans_list_forward_id[i];
		}
		delete[] trans_list_forward_id;
	}
	if (m_raw_intensities)
		delete[] m_raw_intensities;
	if (m_probe_pos)
		delete[] m_probe_pos;
	
}

////////////////////////////////////////////////////////////////////////////////
void CDynProg::set_genestr_len(int32_t genestr_len)
{
	m_genestr_len = genestr_len;
}

int32_t CDynProg::get_num_svms()
{
	return num_svms;
}

void CDynProg::precompute_stop_codons(const char* sequence, int32_t length)
{
	m_genestr_stop.resize_array(length) ;
	m_genestr_stop.zero() ;
	m_genestr_stop.set_name("genestr_stop") ;
	{
	for (int32_t i=0; i<length-2; i++)
		if ((sequence[i]=='t' || sequence[i]=='T') && 
			(((sequence[i+1]=='a' || sequence[i+1]=='A') && 
			  (sequence[i+2]=='a' || sequence[i+2]=='g' || sequence[i+2]=='A' || sequence[i+2]=='G')) ||
			 ((sequence[i+1]=='g'||sequence[i+1]=='G') && (sequence[i+2]=='a' || sequence[i+2]=='A') )))
			m_genestr_stop.element(i)=true ;
		else
			m_genestr_stop.element(i)=false ;
	m_genestr_stop.element(length-1)=false ;
	m_genestr_stop.element(length-1)=false ;
	}
}

void CDynProg::set_num_states(int32_t p_N)
{
	N=p_N ;

	transition_matrix_a_id.resize_array(N,N) ;
	transition_matrix_a.resize_array(N,N) ;
	transition_matrix_a_deriv.resize_array(N,N) ;
	initial_state_distribution_p.resize_array(N) ;
	initial_state_distribution_p_deriv.resize_array(N) ;
	end_state_distribution_q.resize_array(N);
	end_state_distribution_q_deriv.resize_array(N) ;

	m_orf_info.resize_array(N,2) ;
	m_PEN.resize_array(N,N) ;
	m_PEN_state_signals.resize_array(N,1) ;
}

int32_t CDynProg::get_num_states()
{
	return N;
}

void CDynProg::init_tiling_data(
	int32_t* probe_pos, float64_t* intensities, const int32_t num_probes,
	const int32_t seq_len)
{
	m_num_raw_data++;
	m_num_probes_cum[m_num_raw_data] = m_num_probes_cum[m_num_raw_data-1]+num_probes;

	int32_t* tmp_probe_pos = new int32_t[m_num_probes_cum[m_num_raw_data]];
	float64_t* tmp_raw_intensities = new float64_t[m_num_probes_cum[m_num_raw_data]];


	if (m_num_raw_data==1){
		memcpy(tmp_probe_pos, probe_pos, num_probes*sizeof(int32_t));
		memcpy(tmp_raw_intensities, intensities, num_probes*sizeof(float64_t));
		//SG_PRINT("raw_intens:%f \n",*tmp_raw_intensities+2);	
	}else{
		memcpy(tmp_probe_pos, m_probe_pos, m_num_probes_cum[m_num_raw_data-1]*sizeof(int32_t));	
		memcpy(tmp_raw_intensities, m_raw_intensities, m_num_probes_cum[m_num_raw_data-1]*sizeof(float64_t));	
		memcpy(tmp_probe_pos+m_num_probes_cum[m_num_raw_data-1], probe_pos, num_probes*sizeof(int32_t));	
		memcpy(tmp_raw_intensities+m_num_probes_cum[m_num_raw_data-1], intensities, num_probes*sizeof(float64_t));	
	}
	delete[] m_probe_pos;
	delete[] m_raw_intensities;
	m_probe_pos = new int32_t[num_probes];
	m_raw_intensities = new float64_t[num_probes];

	memcpy(m_probe_pos, probe_pos, num_probes*sizeof(int32_t));
	memcpy(m_raw_intensities, intensities, num_probes*sizeof(float64_t));

}

void CDynProg::init_content_svm_value_array(const int32_t p_num_svms, const int32_t seq_len)
{
	m_lin_feat.resize_array(p_num_svms, seq_len);
}
void CDynProg::resize_lin_feat(const int32_t num_new_feat, const int32_t seq_len)
{
	int32_t dim1, dim2;
	m_lin_feat.get_array_size(dim1, dim2);
	ASSERT(dim1==m_num_lin_feat_plifs_cum[m_num_raw_data-1]);
	ASSERT(dim2==seq_len); // == number of candidate positions


	float64_t* arr = m_lin_feat.get_array();
	float64_t* tmp = new float64_t[(dim1+num_new_feat)*dim2];	
	memset(tmp, 0, (dim1+num_new_feat)*dim2*sizeof(float64_t)) ;
	for(int32_t j=0;j<seq_len;j++)
                for(int32_t k=0;k<m_num_lin_feat_plifs_cum[m_num_raw_data-1];k++)
			tmp[j*(dim1+num_new_feat)+k] = arr[j*dim1+k];

	m_lin_feat.set_array(tmp, dim1+num_new_feat,dim2);

	/*for(int32_t j=0;j<5;j++)
	{
		for(int32_t k=0;k<m_num_lin_feat_plifs_cum[m_num_raw_data];k++)
		{
			SG_PRINT("(%i,%i)%f ",k,j,m_lin_feat.get_element(k,j));
		}
		SG_PRINT("\n");
	}
	m_lin_feat.get_array_size(dim1,dim2);
	SG_PRINT("resize_lin_feat: dim1:%i, dim2:%i\n",dim1,dim2);*/

	//SG_PRINT("resize_lin_feat: done\n");
}
//void CDynProg::precompute_tiling_plifs(CPlif** PEN, const INT* tiling_plif_ids, const INT num_tiling_plifs, const INT seq_len, const INT* pos)

void CDynProg::precompute_tiling_plifs(
	CPlif** PEN, const int32_t* tiling_plif_ids,
	const int32_t num_tiling_plifs, const int32_t seq_len, const int32_t* pos)
{
	SG_PRINT("precompute_tiling_plifs:%f num_tiling_plifs:%i\n",m_raw_intensities[0], num_tiling_plifs);

	/*int32_t tiling_plif_ids[num_svms];
	int32_t num = 0;
        for (int32_t i=0; i<num_penalties; i++)
	{
		CPlif * plif = PEN[i];
		if (plif->get_use_svm()>num_svms)
		{
			tiling_plif_ids[num] = i;
			num++;
		}
	}*/
	m_num_lin_feat_plifs_cum[m_num_raw_data] = m_num_lin_feat_plifs_cum[m_num_raw_data-1]+ num_tiling_plifs;
	//for (INT d=0;d<=m_num_raw_data;d++)
	//	SG_PRINT("lin_feat_plifs%i: %i\n",d,m_num_lin_feat_plifs_cum[d]);
	float64_t* tiling_plif = new float64_t[num_tiling_plifs];
	float64_t* svm_value = new float64_t[m_num_lin_feat_plifs_cum[m_num_raw_data]];
	for (int32_t i=0; i<m_num_lin_feat_plifs_cum[m_num_raw_data]; i++)
		svm_value[i]=0.0;
	int32_t tiling_rows[num_tiling_plifs];
	for (int32_t i=0; i<num_tiling_plifs; i++)
	{
		tiling_plif[i]=0.0;
		CPlif * plif = PEN[tiling_plif_ids[i]];
		tiling_rows[i] = plif->get_use_svm();
		//float64_t* limits = plif->get_plif_limits();
		//for(int32_t j=0;j<20;j++)
		//	SG_PRINT("%.2f, ",limits[j]);
		//SG_PRINT("\n ");
	//	SG_PRINT("tiling_rows[%i]:%i, tiling_plif_ids[%i]:%i  \n",i,tiling_rows[i],i,tiling_plif_ids[i]);
		ASSERT(tiling_rows[i]-1==m_num_lin_feat_plifs_cum[m_num_raw_data-1]+i)
	}
	resize_lin_feat(num_tiling_plifs, seq_len);


	int32_t* p_tiling_pos  = &m_probe_pos[m_num_probes_cum[m_num_raw_data-1]];
	float64_t* p_tiling_data = &m_raw_intensities[m_num_probes_cum[m_num_raw_data-1]];
	int32_t num=m_num_probes_cum[m_num_raw_data-1];

	for (int32_t pos_idx=0;pos_idx<seq_len;pos_idx++)
	{
		//SG_PRINT("pos[%i]: %i  \n",pos_idx,pos[pos_idx]);
		//SG_PRINT("*p_tiling_pos: %i  \n",*p_tiling_pos);
		while (num<m_num_probes_cum[m_num_raw_data]&&*p_tiling_pos<pos[pos_idx])
		{
			//SG_PRINT("raw_intens: %f  \n",*p_tiling_data);
			for (int32_t i=0; i<num_tiling_plifs; i++)
			{
				svm_value[m_num_lin_feat_plifs_cum[m_num_raw_data-1]+i]=*p_tiling_data;
				//if (svm_value[m_num_lin_feat+i]>15||svm_value[m_num_lin_feat+i]<4)
				//	SG_PRINT("uninitialized value, value:%f, i:%i, pos:%i \n", svm_value[m_num_lin_feat+i], i, pos[pos_idx]);
				CPlif * plif = PEN[tiling_plif_ids[i]];
				//SG_PRINT("m_num_lin_feat_plifs_cum[m_num_raw_data]:%i, i:%i, plif->get_use_svm():%i\n",m_num_lin_feat_plifs_cum[m_num_raw_data],i, plif->get_use_svm());
				ASSERT(m_num_lin_feat_plifs_cum[m_num_raw_data-1]+i==plif->get_use_svm()-1);
				plif->set_do_calc(true);
				tiling_plif[i]+=plif->lookup_penalty(0,svm_value);
				//SG_PRINT("true: plif->lookup_penalty: %f  tiling_plif[%i]:%f \n",plif->lookup_penalty(0,svm_value),i,tiling_plif[i]);
				plif->set_do_calc(false);
				//SG_PRINT("false: plif->lookup_penalty: %f  \n",plif->lookup_penalty(0,svm_value));
			}
			//SG_PRINT("p_tiling_data:%f\n",*p_tiling_data);
			p_tiling_data++;
			p_tiling_pos++;
			num++;
		}
		for (int32_t i=0; i<num_tiling_plifs; i++)
			m_lin_feat.set_element(tiling_plif[i],tiling_rows[i]-1,pos_idx);
	}
	delete[] svm_value;
	delete[] tiling_plif;
}

void CDynProg::create_word_string(
	const char* genestr, int32_t genestr_num, int32_t genestr_len,
	uint16_t*** wordstr)
{
	for (int32_t k=0; k<genestr_num; k++)
	{
		wordstr[k]=new uint16_t*[num_degrees] ;
		for (int32_t j=0; j<num_degrees; j++)
		{
			wordstr[k][j]=NULL ;
			{
				wordstr[k][j]=new uint16_t[genestr_len] ;
				for (int32_t i=0; i<genestr_len; i++)
					switch (genestr[i])
					{
					case 'A':
					case 'a': wordstr[k][j][i]=0 ; break ;
					case 'C':
					case 'c': wordstr[k][j][i]=1 ; break ;
					case 'G':
					case 'g': wordstr[k][j][i]=2 ; break ;
					case 'T':
					case 't': wordstr[k][j][i]=3 ; break ;
					default: ASSERT(0) ;
					}
				translate_from_single_order(wordstr[k][j], genestr_len, word_degree[j]-1, word_degree[j]) ;
			}
		}
	}
	//precompute_stop_codons(genestr, genestr_len);
}

void CDynProg::precompute_content_values(
	uint16_t*** wordstr, const int32_t *pos,const int32_t seq_len,
	const int32_t genestr_len,float64_t *dictionary_weights,int32_t dict_len)
{
	//SG_PRINT("seq_len=%i, genestr_len=%i, dict_len=%i, num_svms=%i, num_degrees=%i\n",seq_len, genestr_len, dict_len, num_svms, num_degrees);

	dict_weights.set_array(dictionary_weights, dict_len, num_svms, false, false) ;
	dict_weights_array=dict_weights.get_array() ;

	//int32_t d1 = mod_words.get_dim1();
	//int32_t d2 = mod_words.get_dim2();
	//SG_PRINT("d1:%i, d2:%i \n",d1, d2);
	//for (int32_t p=0 ; p<d1 ; p++)
	//{
	//	for (int32_t q=0 ; q<d2 ; q++)
	//		SG_PRINT("%i ",mod_words.get_element(p,q));
	//	SG_PRINT("\n");
	//}

	for (int32_t p=0 ; p<seq_len-1 ; p++)
	{
		int32_t from_pos = pos[p];
		int32_t to_pos = pos[p+1];
		float64_t my_svm_values_unnormalized[num_svms] ;
		//SG_PRINT("%i(%i->%i) ",p,from_pos, to_pos);
      
		ASSERT(from_pos<=genestr_len)	
		ASSERT(to_pos<=genestr_len)	
			
		for (int32_t s=0; s<num_svms; s++)
		{
			my_svm_values_unnormalized[s]=0.0;//precomputed_svm_values.element(s,p);
		}
		for (int32_t i=from_pos; i<to_pos;i++)
                {
			for (int32_t j=0; j<num_degrees; j++)
			{
				uint16_t word = wordstr[0][j][i] ;
				for (int32_t s=0; s<num_svms; s++)
				{
					// check if this k-mere should be considered for this SVM
					if (mod_words.get_element(s,0)==3 && i%3!=mod_words.get_element(s,1))
						continue;
					my_svm_values_unnormalized[s] += dict_weights_array[(word+cum_num_words_array[j])+s*cum_num_words_array[num_degrees]] ;
				}
			}
		}
		for (int32_t s=0; s<num_svms; s++)
		{
			float64_t prev = m_lin_feat.get_element(s,p);
			m_lin_feat.set_element(prev + my_svm_values_unnormalized[s],s,p+1);

			ASSERT(prev>-1e20);
		}
	}
	for (int32_t j=0; j<num_degrees; j++)
		delete[] wordstr[0][j] ;
	delete[] wordstr[0] ;
}

void CDynProg::set_p_vector(float64_t *p, int32_t p_N)
{
	if (!(p_N==N))
		SG_ERROR("length of start prob vector p (%i) is not equal to the number of states (%i), p_N: %i\n",p_N, N);
	//m_orf_info.resize_array(p_N,2);
	//m_PEN.resize_array(p_N,p_N);

	initial_state_distribution_p.set_array(p, p_N, true, true);
}

void CDynProg::set_q_vector(float64_t *q, int32_t q_N)
{
	if (!(q_N==N))
		SG_ERROR("length of end prob vector q (%i) is not equal to the number of states (%i), p_N: %i\n",q_N, N);
	end_state_distribution_q.set_array(q, q_N, true, true);
}

void CDynProg::set_a(float64_t *a, int32_t p_M, int32_t p_N)
{
	ASSERT(p_N==N);
	ASSERT(p_M==p_N);
	transition_matrix_a.set_array(a, p_N, p_N, true, true);
	transition_matrix_a_deriv.resize_array(p_N, p_N);
}

void CDynProg::set_a_id(int32_t *a, int32_t p_M, int32_t p_N)
{
	ASSERT(p_N==N);
	ASSERT(p_M==p_N);
	transition_matrix_a_id.set_array(a, p_N, p_N, true, true);
	max_a_id = 0;
	for (int32_t i=0; i<p_N; i++)
		for (int32_t j=0; j<p_N; j++)
			max_a_id=CMath::max(max_a_id, transition_matrix_a_id.element(i,j));
}

void CDynProg::set_a_trans_matrix(
	float64_t *a_trans, int32_t num_trans, int32_t p_N)
{
	if (!((p_N==3) || (p_N==4)))
		SG_ERROR("!((p_N==3) || (p_N==4)), p_N: %i\n",p_N);

	delete[] trans_list_forward ;
	delete[] trans_list_forward_cnt ;
	delete[] trans_list_forward_val ;
	delete[] trans_list_forward_id ;

	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_len = 0 ;

	transition_matrix_a.zero() ;
	transition_matrix_a_id.zero() ;

	mem_initialized = true ;

	trans_list_forward_cnt=NULL ;
	trans_list_len = N ;
	trans_list_forward = new T_STATES*[N] ;
	trans_list_forward_cnt = new T_STATES[N] ;
	trans_list_forward_val = new float64_t*[N] ;
	trans_list_forward_id = new int32_t*[N] ;
	
	int32_t start_idx=0;
	for (int32_t j=0; j<N; j++)
	{
		int32_t old_start_idx=start_idx;

		while (start_idx<num_trans && a_trans[start_idx+num_trans]==j)
		{
			start_idx++;
			
			if (start_idx>1 && start_idx<num_trans)
				ASSERT(a_trans[start_idx+num_trans-1] <= a_trans[start_idx+num_trans]);
		}
		
		if (start_idx>1 && start_idx<num_trans)
			ASSERT(a_trans[start_idx+num_trans-1] <= a_trans[start_idx+num_trans]);
		
		int32_t len=start_idx-old_start_idx;
		ASSERT(len>=0);
		
		trans_list_forward_cnt[j] = 0 ;
		
		if (len>0)
		{
			trans_list_forward[j]     = new T_STATES[len] ;
			trans_list_forward_val[j] = new float64_t[len] ;
			trans_list_forward_id[j] = new int32_t[len] ;
		}
		else
		{
			trans_list_forward[j]     = NULL;
			trans_list_forward_val[j] = NULL;
			trans_list_forward_id[j]  = NULL;
		}
	}
	
	for (int32_t i=0; i<num_trans; i++)
	{
		int32_t from_state   = (int32_t)a_trans[i] ;
		int32_t to_state = (int32_t)a_trans[i+num_trans] ;
		float64_t val = a_trans[i+num_trans*2] ;
		int32_t id = 0 ;
		if (p_N==4)
			id = (int32_t)a_trans[i+num_trans*3] ;
		//SG_DEBUG( "id=%i\n", id) ;
			
		ASSERT(to_state>=0 && to_state<N);
		ASSERT(from_state>=0 && from_state<N);
		
		trans_list_forward[to_state][trans_list_forward_cnt[to_state]]=from_state ;
		trans_list_forward_val[to_state][trans_list_forward_cnt[to_state]]=val ;
		trans_list_forward_id[to_state][trans_list_forward_cnt[to_state]]=id ;
		trans_list_forward_cnt[to_state]++ ;
		transition_matrix_a.element(from_state, to_state) = val ;
		transition_matrix_a_id.element(from_state, to_state) = id ;
		//SG_PRINT("from_state:%i to_state:%i trans_matrix_a_id:%i \n",from_state, to_state,transition_matrix_a_id.element(from_state, to_state));
	} ;

	max_a_id = 0 ;
	for (int32_t i=0; i<N; i++)
		for (int32_t j=0; j<N; j++)
		{
			//if (transition_matrix_a_id.element(i,j))
			//SG_DEBUG( "(%i,%i)=%i\n", i,j, transition_matrix_a_id.element(i,j)) ;
			max_a_id = CMath::max(max_a_id, transition_matrix_a_id.element(i,j)) ;
		}
	//SG_DEBUG( "max_a_id=%i\n", max_a_id) ;
}

void CDynProg::init_svm_arrays(int32_t p_num_degrees, int32_t p_num_svms)
{
	svm_arrays_clean=false ;

	word_degree.resize_array(num_degrees) ;

	cum_num_words.resize_array(num_degrees+1) ;
	cum_num_words_array=cum_num_words.get_array() ;

	num_words.resize_array(num_degrees) ;
	num_words_array=num_words.get_array() ;
	
	//svm_values_unnormalized.resize_array(num_degrees, num_svms) ;
	svm_pos_start.resize_array(num_degrees) ;
	num_unique_words.resize_array(num_degrees) ;
} 


void CDynProg::init_word_degree_array(
	int32_t * p_word_degree_array, int32_t num_elem)
{
	svm_arrays_clean=false ;

	word_degree.resize_array(num_degrees) ;
	ASSERT(num_degrees==num_elem);


	for (int32_t i=0; i<num_degrees; i++)
		word_degree[i]=p_word_degree_array[i] ;

} 

void CDynProg::init_cum_num_words_array(
	int32_t * p_cum_num_words_array, int32_t num_elem)
{
	svm_arrays_clean=false ;

	cum_num_words.resize_array(num_degrees+1) ;
	cum_num_words_array=cum_num_words.get_array() ;
	ASSERT(num_degrees+1==num_elem);

	for (int32_t i=0; i<num_degrees+1; i++)
		cum_num_words[i]=p_cum_num_words_array[i] ;
} 

void CDynProg::init_num_words_array(
	int32_t * p_num_words_array, int32_t num_elem)
{
	svm_arrays_clean=false ;

	num_words.resize_array(num_degrees) ;
	num_words_array=num_words.get_array() ;
	ASSERT(num_degrees==num_elem);

	for (int32_t i=0; i<num_degrees; i++)
		num_words[i]=p_num_words_array[i] ;

	//word_used.resize_array(num_degrees, num_words[num_degrees-1], num_strings) ;
	//word_used_array=word_used.get_array() ;
} 

void CDynProg::init_mod_words_array(
	int32_t * mod_words_input, int32_t num_elem, int32_t num_columns)
{
	//for (int32_t i=0; i<num_columns; i++)
	//{
	//	for (int32_t j=0; j<num_elem; j++)
	//		SG_PRINT("%i ",mod_words_input[i*num_elem+j]);
	//	SG_PRINT("\n");
	//}
	svm_arrays_clean=false ;

	ASSERT(num_svms==num_elem);
	ASSERT(num_columns==2);

	mod_words.set_array(mod_words_input, num_elem, 2, true, true) ;
	mod_words_array = mod_words.get_array() ;
	
	/*SG_DEBUG( "mod_words=[") ;
	for (int32_t i=0; i<num_elem; i++)
		SG_DEBUG( "%i, ", p_mod_words_array[i]) ;
		SG_DEBUG( "]\n") ;*/
} 

void CDynProg::init_sign_words_array(bool* p_sign_words_array, int32_t num_elem)
{
	svm_arrays_clean=false ;

	ASSERT(num_svms==num_elem);

	sign_words.set_array(p_sign_words_array, num_elem, true, true) ;
	sign_words_array = sign_words.get_array() ;
} 

void CDynProg::init_string_words_array(
	int32_t* p_string_words_array, int32_t num_elem)
{
	svm_arrays_clean=false ;

	ASSERT(num_svms==num_elem);

	string_words.set_array(p_string_words_array, num_elem, true, true) ;
	string_words_array = string_words.get_array() ;
} 

bool CDynProg::check_svm_arrays()
{
	//SG_DEBUG( "wd_dim1=%d, cum_num_words=%d, num_words=%d, svm_pos_start=%d, num_uniq_w=%d, mod_words_dims=(%d,%d), sign_w=%d,string_w=%d\n num_degrees=%d, num_svms=%d, num_strings=%d", word_degree.get_dim1(), cum_num_words.get_dim1(), num_words.get_dim1(), svm_pos_start.get_dim1(), num_unique_words.get_dim1(), mod_words.get_dim1(), mod_words.get_dim2(), sign_words.get_dim1(), string_words.get_dim1(), num_degrees, num_svms, num_strings);
	if ((word_degree.get_dim1()==num_degrees) &&
			(cum_num_words.get_dim1()==num_degrees+1) &&
			(num_words.get_dim1()==num_degrees) &&
			//(word_used.get_dim1()==num_degrees) &&
			//(word_used.get_dim2()==num_words[num_degrees-1]) &&
			//(word_used.get_dim3()==num_strings) &&
			//		(svm_values_unnormalized.get_dim1()==num_degrees) &&
			//		(svm_values_unnormalized.get_dim2()==num_svms) &&
			(svm_pos_start.get_dim1()==num_degrees) &&
			(num_unique_words.get_dim1()==num_degrees) &&
			(mod_words.get_dim1()==num_svms) &&
			(mod_words.get_dim2()==2) && 
			(sign_words.get_dim1()==num_svms) &&
			(string_words.get_dim1()==num_svms))
	{
		svm_arrays_clean=true ;
		return true ;
	}
	else
	{
		if ((num_unique_words.get_dim1()==num_degrees) &&
            (mod_words.get_dim1()==num_svms) &&
			(mod_words.get_dim2()==2) &&
			(sign_words.get_dim1()==num_svms) &&
            (string_words.get_dim1()==num_svms))
			SG_PRINT("OK\n") ;
		else
			SG_PRINT("not OK\n") ;

		if (!(word_degree.get_dim1()==num_degrees))
			SG_WARNING("SVM array: word_degree.get_dim1()!=num_degrees") ;
		if (!(cum_num_words.get_dim1()==num_degrees+1))
			SG_WARNING("SVM array: cum_num_words.get_dim1()!=num_degrees+1") ;
		if (!(num_words.get_dim1()==num_degrees))
			SG_WARNING("SVM array: num_words.get_dim1()==num_degrees") ;
		if (!(svm_pos_start.get_dim1()==num_degrees))
			SG_WARNING("SVM array: svm_pos_start.get_dim1()!=num_degrees") ;
		if (!(num_unique_words.get_dim1()==num_degrees))
			SG_WARNING("SVM array: num_unique_words.get_dim1()!=num_degrees") ;
		if (!(mod_words.get_dim1()==num_svms))
			SG_WARNING("SVM array: mod_words.get_dim1()!=num_svms") ;
		if (!(mod_words.get_dim2()==2))
			SG_WARNING("SVM array: mod_words.get_dim2()!=2") ;
		if (!(sign_words.get_dim1()==num_svms))
			SG_WARNING("SVM array: sign_words.get_dim1()!=num_svms") ;
		if (!(string_words.get_dim1()==num_svms))
			SG_WARNING("SVM array: string_words.get_dim1()!=num_svms") ;

		svm_arrays_clean=false ;
		return false ;	
	}
}

void CDynProg::best_path_set_seq(float64_t *seq, int32_t p_N, int32_t seq_len)
{
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	} ;

	ASSERT(p_N==N);
	ASSERT(initial_state_distribution_p.get_dim1()==N);
	ASSERT(end_state_distribution_q.get_dim1()==N);
	
	m_seq.set_array(seq, N, seq_len, 1, true, true) ;
	this->N=N ;

	m_call=3 ;
	m_step=2 ;
}

void CDynProg::best_path_set_seq3d(
	float64_t *seq, int32_t p_N, int32_t seq_len, int32_t max_num_signals)
{
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	} ;

	ASSERT(p_N==N);
	ASSERT(initial_state_distribution_p.get_dim1()==N);
	ASSERT(end_state_distribution_q.get_dim1()==N);
	
	m_seq.set_array(seq, N, seq_len, max_num_signals, true, true) ;
	this->N=N ;

	m_call=3 ;
	m_step=2 ;
}

void CDynProg::best_path_set_pos(int32_t *pos, int32_t seq_len)  
{
	if (m_step!=2)
		SG_ERROR( "please call best_path_set_seq first\n") ;
	
	if (seq_len!=m_seq.get_dim2())
		SG_ERROR( "pos size does not match previous info %i!=%i\n", seq_len, m_seq.get_dim2()) ;

	m_pos.set_array(pos, seq_len, true, true) ;

	m_step=3 ;
}

void CDynProg::best_path_set_orf_info(int32_t *orf_info, int32_t m, int32_t n) 
{
	//if (m_step!=3)
	//	SG_ERROR( "please call best_path_set_pos first\n") ;
		
	//if (m!=N)
	//	SG_ERROR( "orf_info size does not match previous info %i!=%i\n", m, N) ;
	if (n!=2)
		SG_ERROR( "orf_info size incorrect %i!=2\n", n) ;
	m_orf_info.set_array(orf_info, m, n, true, true) ;

	m_call=1 ;
	m_step=4 ;
}

void CDynProg::best_path_set_segment_sum_weights(
	float64_t *segment_sum_weights, int32_t num_states, int32_t seq_len)
{
	if (m_step!=3)
		SG_ERROR( "please call best_path_set_pos first\n") ;
		
	if (num_states!=N)
		SG_ERROR( "segment_sum_weights size does not match previous info %i!=%i\n", num_states, N) ;
	if (seq_len!=m_pos.get_dim1())
		SG_ERROR( "segment_sum_weights size incorrect %i!=%i\n", seq_len, m_pos.get_dim1()) ;

	m_segment_sum_weights.set_array(segment_sum_weights, num_states, seq_len, true, true) ;
	
	m_call=2 ;
	m_step=4 ;
}

void CDynProg::best_path_set_plif_list(CDynamicArray<CPlifBase*>* plifs)
{
	ASSERT(plifs);
	CPlifBase** plif_list=plifs->get_array();
	int32_t num_plif=plifs->get_num_elements();

	if (m_step!=4)
		SG_ERROR( "please call best_path_set_orf_info or best_path_segment_sum_weights first\n") ;

	m_plif_list.set_array(plif_list, num_plif, true, true) ;

	m_step=5 ;
}

void CDynProg::best_path_set_plif_id_matrix(
	int32_t *plif_id_matrix, int32_t m, int32_t n)
{
	if (m_step!=5)
		SG_ERROR( "please call best_path_set_plif_list first\n") ;

	if ((m!=N) || (n!=N))
		SG_ERROR( "plif_id_matrix size does not match previous info %i!=%i or %i!=%i\n", m, N, n, N) ;

	CArray2<int32_t> id_matrix(plif_id_matrix, N, N, false, false) ;
#ifdef DYNPROG_DEBUG
	id_matrix.set_name("id_matrix");
	id_matrix.display_array();
#endif //DYNPROG_DEBUG
	m_PEN.resize_array(N, N) ;
	for (int32_t i=0; i<N; i++)
		for (int32_t j=0; j<N; j++)
			if (id_matrix.element(i,j)>=0)
				m_PEN.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN.element(i,j)=NULL ;

	m_step=6 ;
}

void CDynProg::best_path_set_plif_state_signal_matrix(
	int32_t *plif_id_matrix, int32_t m, int32_t max_num_signals)
{
	if (m_step!=6)
		SG_ERROR( "please call best_path_set_plif_id_matrix first\n") ;
	
	if (m!=N)
		SG_ERROR( "plif_state_signal_matrix size does not match previous info %i!=%i\n", m, N) ;

	if (m_seq.get_dim3() != max_num_signals)
		SG_ERROR( "size(plif_state_signal_matrix,2) does not match with size(m_seq,3): %i!=%i\nSorry, Soeren... interface changed\n", m_seq.get_dim3(), max_num_signals) ;

	CArray2<int32_t> id_matrix(plif_id_matrix, N, max_num_signals, false, false) ;
	m_PEN_state_signals.resize_array(N, max_num_signals) ;
	for (int32_t i=0; i<N; i++)
	{
		for (int32_t j=0; j<max_num_signals; j++)
		{
			if (id_matrix.element(i,j)>=0)
				m_PEN_state_signals.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN_state_signals.element(i,j)=NULL ;
		}
	}

	m_step=6 ;
}

void CDynProg::best_path_set_genestr(
	char* genestr, int32_t genestr_len, int32_t genestr_num)
{
	if (m_step!=6)
		SG_ERROR( "please call best_path_set_plif_id_matrix first\n") ;

	ASSERT(genestr);
	ASSERT(genestr_len>0);
	ASSERT(genestr_num>0);

	m_genestr.set_array(genestr, genestr_len, genestr_num, true, true) ;

	m_step=7 ;
}

void CDynProg::best_path_set_my_state_seq(
	int32_t* my_state_seq, int32_t seq_len)
{
	ASSERT(my_state_seq && seq_len>0);
	m_my_state_seq.resize_array(seq_len);
	for (int32_t i=0; i<seq_len; i++)
		m_my_state_seq[i]=my_state_seq[i];
}

void CDynProg::best_path_set_my_pos_seq(int32_t* my_pos_seq, int32_t seq_len)
{
	ASSERT(my_pos_seq && seq_len>0);
	m_my_pos_seq.resize_array(seq_len);
	for (int32_t i=0; i<seq_len; i++)
		m_my_pos_seq[i]=my_pos_seq[i];
}

void CDynProg::best_path_set_dict_weights(
	float64_t* dictionary_weights, int32_t dict_len, int32_t n)
{
	if (m_step!=7)
		SG_ERROR( "please call best_path_set_genestr first\n") ;

	if (num_svms!=n)
		SG_ERROR( "dict_weights array does not match num_svms=%i!=%i\n", num_svms, n) ;

	m_dict_weights.set_array(dictionary_weights, dict_len, num_svms, true, true) ;

	// initialize, so it does not bother when not used
	m_segment_loss.resize_array(max_a_id+1, max_a_id+1, 2) ;
	m_segment_loss.zero() ;
	m_segment_ids.resize_array(m_seq.get_dim2()) ;
	m_segment_mask.resize_array(m_seq.get_dim2()) ;
	m_segment_ids.zero() ;
	m_segment_mask.zero() ;

	m_step=8 ;
}

void CDynProg::best_path_set_segment_loss(
	float64_t* segment_loss, int32_t m, int32_t n)
{
	// here we need two matrices. Store it in one: 2N x N
	if (2*m!=n)
		SG_ERROR( "segment_loss should be 2 x quadratic matrix: %i!=%i\n", 2*m, n) ;

	if (m!=max_a_id+1)
		SG_ERROR( "segment_loss size should match max_a_id: %i!=%i\n", m, max_a_id+1) ;

	m_segment_loss.set_array(segment_loss, m, n/2, 2, true, true) ;
	/*for (int32_t i=0; i<n; i++)
		for (int32_t j=0; j<n; j++)
		SG_DEBUG( "loss(%i,%i)=%f\n", i,j, m_segment_loss.element(0,i,j)) ;*/
}

void CDynProg::best_path_set_segment_ids_mask(
	int32_t* segment_ids, float64_t* segment_mask, int32_t m)
{
	int32_t max_id = 0;
	for (int32_t i=1;i<m;i++)
		max_id = CMath::max(max_id,segment_ids[i]);
	SG_PRINT("max_id: %i, m:%i\n",max_id, m); 	
	m_segment_ids.set_array(segment_ids, m, false, true) ;
	m_segment_ids.set_name("m_segment_ids");
	m_segment_mask.set_array(segment_mask, m, false, true) ;
	m_segment_mask.set_name("m_segment_mask");
}


void CDynProg::best_path_call(int32_t nbest, bool use_orf) 
{
	if (m_step!=8)
		SG_ERROR( "please call best_path_set_dict_weights first\n") ;
	if (m_call!=1)
		SG_ERROR( "please call best_path_set_orf_info first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;
	ASSERT(m_seq.get_dim2()==m_pos.get_dim1()) ;

	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;
	m_positions.resize_array(nbest, m_seq.get_dim2()) ;

	m_call=1 ;

	ASSERT(nbest==1||nbest==2) ;
	ASSERT(m_genestr.get_dim2()==1) ;
	if (nbest==1)
		best_path_trans<1,false,false>(m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), 
								m_orf_info.get_array(), m_PEN.get_array(),
								m_PEN_state_signals.get_array(), m_PEN_state_signals.get_dim2(),
								m_genestr.get_dim2(), m_scores.get_array(), 
								m_states.get_array(), m_positions.get_array(),
								use_orf) ;
	else
		best_path_trans<2,false,false>(m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), 
								m_orf_info.get_array(), m_PEN.get_array(),
								m_PEN_state_signals.get_array(), m_PEN_state_signals.get_dim2(),
								m_genestr.get_dim2(), m_scores.get_array(),
								m_states.get_array(), m_positions.get_array(),
								use_orf) ;

	m_step=9 ;
}

void CDynProg::best_path_deriv_call() 
{
	//if (m_step!=8)
		//SG_ERROR( "please call best_path_set_dict_weights first\n") ;
	//if (m_call!=1)
		//SG_ERROR( "please call best_path_set_orf_info first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;
	ASSERT(m_seq.get_dim2()==m_pos.get_dim1()) ;

	m_call=5 ; // or so ...

	m_my_scores.resize_array(m_my_state_seq.get_array_size()) ;
	m_my_losses.resize_array(m_my_state_seq.get_array_size()) ;

	best_path_trans_deriv(m_my_state_seq.get_array(), m_my_pos_seq.get_array(), 
						  m_my_scores.get_array(), m_my_losses.get_array(), m_my_state_seq.get_array_size(),
						  m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), 
						  m_PEN.get_array(), m_PEN_state_signals.get_array(), m_PEN_state_signals.get_dim2(), m_genestr.get_dim2()) ;

	m_step=12 ;
}


void CDynProg::best_path_2struct_call(int32_t nbest) 
{
	if (m_step!=8)
		SG_ERROR( "please call best_path_set_orf_dict_weights first\n") ;
	if (m_call!=2)
		SG_ERROR( "please call best_path_set_segment_sum_weights first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;
	ASSERT(m_seq.get_dim2()==m_pos.get_dim1()) ;
	
	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;
	m_positions.resize_array(nbest, m_seq.get_dim2()) ;

	m_call=2 ;

	best_path_2struct(m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), 
					  m_PEN.get_array(), 
					  m_genestr.get_array(), m_genestr.get_dim1(),
					  nbest, 
					  m_scores.get_array(), m_states.get_array(), m_positions.get_array(),
					  m_dict_weights.get_array(), m_dict_weights.get_dim1()*m_dict_weights.get_dim2(), 
					  m_segment_sum_weights.get_array()) ;

	m_step=9 ;
}

void CDynProg::best_path_simple_call(int32_t nbest) 
{
	if (m_step!=2)
		SG_ERROR( "please call best_path_set_seq first\n") ;
	if (m_call!=3)
		SG_ERROR( "please call best_path_set_seq first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;

	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;

	m_call=3 ;

	best_path_trans_simple(m_seq.get_array(), m_seq.get_dim2(), 
						   nbest, 
						   m_scores.get_array(), m_states.get_array()) ;
	
	m_step=9 ;
}

void CDynProg::best_path_deriv_call(int32_t nbest)
{
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	} ;

	//FIXME
}


void CDynProg::best_path_get_scores(float64_t **scores, int32_t *m) 
{
	if (m_step!=9 && m_step!=12)
		SG_ERROR( "please call best_path*_call first\n") ;

	if (m_step==9)
	{
		*scores=m_scores.get_array() ;
		*m=m_scores.get_dim1() ;
	} else
	{
		*scores=m_my_scores.get_array() ;
		*m=m_my_scores.get_dim1() ;
	}

	m_step=10 ;
}

void CDynProg::best_path_get_states(int32_t **states, int32_t *m, int32_t *n) 
{
	if (m_step!=10)
		SG_ERROR( "please call best_path_get_score first\n") ;
	
	*states=m_states.get_array() ;
	*m=m_states.get_dim1() ;
	*n=m_states.get_dim2() ;

	m_step=11 ;
}

void CDynProg::best_path_get_positions(
	int32_t **positions, int32_t *m, int32_t *n)
{
	if (m_step!=11)
		SG_ERROR( "please call best_path_get_positions first\n") ;
	if (m_call==3)
		SG_ERROR( "no position information for best_path_simple\n") ;
	
	*positions=m_positions.get_array() ;
	*m=m_positions.get_dim1() ;
	*n=m_positions.get_dim2() ;
}

void CDynProg::best_path_get_losses(float64_t** losses, int32_t* seq_len)
{
	ASSERT(losses && seq_len);
	*losses=m_my_losses.get_array();
	*seq_len=m_my_losses.get_dim1();
}


////////////////////////////////////////////////////////////////////////////////

float64_t CDynProg::best_path_no_b(
	int32_t max_iter, int32_t &best_iter, int32_t *my_path)
{
	CArray2<T_STATES> psi(max_iter, N) ;
	CArray<float64_t>* delta = new CArray<float64_t>(N) ;
	CArray<float64_t>* delta_new = new CArray<float64_t>(N) ;
	
	{ // initialization
		for (int32_t i=0; i<N; i++)
		{
			delta->element(i) = get_p(i) ;
			psi.element(0, i)= 0 ;
		}
	} 
	
	float64_t best_iter_prob = CMath::ALMOST_NEG_INFTY ;
	best_iter = 0 ;
	
	// recursion
	for (int32_t t=1; t<max_iter; t++)
	{
		CArray<float64_t>* dummy;
		int32_t NN=N ;
		for (int32_t j=0; j<NN; j++)
		{
			float64_t maxj = delta->element(0) + transition_matrix_a.element(0,j);
			int32_t argmax=0;
			
			for (int32_t i=1; i<NN; i++)
			{
				float64_t temp = delta->element(i) + transition_matrix_a.element(i,j);
				
				if (temp>maxj)
				{
					maxj=temp;
					argmax=i;
				}
			}
			delta_new->element(j)=maxj ;
			psi.element(t, j)=argmax ;
		}
		
		dummy=delta;
		delta=delta_new;
		delta_new=dummy;	//switch delta/delta_new
		
		{ //termination
			float64_t maxj=delta->element(0)+get_q(0);
			int32_t argmax=0;
			
			for (int32_t i=1; i<N; i++)
			{
				float64_t temp=delta->element(i)+get_q(i);
				
				if (temp>maxj)
				{
					maxj=temp;
					argmax=i;
				}
			}
			//pat_prob=maxj;
			
			if (maxj>best_iter_prob)
			{
				my_path[t]=argmax;
				best_iter=t ;
				best_iter_prob = maxj ;
			} ;
		} ;
	}

	
	{ //state sequence backtracking
		for (int32_t t = best_iter; t>0; t--)
		{
			my_path[t-1]=psi.element(t, my_path[t]);
		}
	}

	delete delta ;
	delete delta_new ;
	
	return best_iter_prob ;
}

void CDynProg::best_path_no_b_trans(
	int32_t max_iter, int32_t &max_best_iter, int16_t nbest,
	float64_t *prob_nbest, int32_t *my_paths)
{
	//T_STATES *psi=new T_STATES[max_iter*N*nbest] ;
	CArray3<T_STATES> psi(max_iter, N, nbest) ;
	CArray3<int16_t> ktable(max_iter, N, nbest) ;
	CArray2<int16_t> ktable_ends(max_iter, nbest) ;

	CArray<float64_t> tempvv(nbest*N) ;
	CArray<int32_t> tempii(nbest*N) ;

	CArray2<T_STATES> path_ends(max_iter, nbest) ;
	CArray2<float64_t> *delta=new CArray2<float64_t>(N, nbest) ;
	CArray2<float64_t> *delta_new=new CArray2<float64_t>(N, nbest) ;
	CArray2<float64_t> delta_end(max_iter, nbest) ;

	CArray2<int32_t> paths(max_iter, nbest) ;
	paths.set_array(my_paths, max_iter, nbest, false, false) ;

	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			delta->element(i,0) = get_p(i) ;
			for (int16_t k=1; k<nbest; k++)
			{
				delta->element(i,k)=-CMath::INFTY ;
				ktable.element(0,i,k)=0 ;
			}
		}
	}
	
	// recursion
	for (int32_t t=1; t<max_iter; t++)
	{
		CArray2<float64_t>* dummy=NULL;

		for (T_STATES j=0; j<N; j++)
		{
			const T_STATES num_elem   = trans_list_forward_cnt[j] ;
			const T_STATES *elem_list = trans_list_forward[j] ;
			const float64_t *elem_val = trans_list_forward_val[j] ;
			
			int32_t list_len=0 ;
			for (int16_t diff=0; diff<nbest; diff++)
			{
				for (int32_t i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					
					tempvv.element(list_len) = -(delta->element(ii,diff) + elem_val[i]) ;
					tempii.element(list_len) = diff*N + ii ;
					list_len++ ;
				}
			}
			CMath::qsort_index(tempvv.get_array(), tempii.get_array(), list_len) ;
			
			for (int16_t k=0; k<nbest; k++)
			{
				if (k<list_len)
				{
					delta_new->element(j,k)  = -tempvv[k] ;
					psi.element(t,j,k)      = (tempii[k]%N) ;
					ktable.element(t,j,k)   = (tempii[k]-(tempii[k]%N))/N ;
				}
				else
				{
					delta_new->element(j,k)  = -CMath::INFTY ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)   = 0 ;
				}
			}
		}
		
		dummy=delta;
		delta=delta_new;
		delta_new=dummy;	//switch delta/delta_new
		
		{ //termination
			int32_t list_len = 0 ;
			for (int16_t diff=0; diff<nbest; diff++)
			{
				for (T_STATES i=0; i<N; i++)
				{
					tempvv.element(list_len) = -(delta->element(i,diff)+get_q(i));
					tempii.element(list_len) = diff*N + i ;
					list_len++ ;
				}
			}
			CMath::qsort_index(tempvv.get_array(), tempii.get_array(), list_len) ;
			
			for (int16_t k=0; k<nbest; k++)
			{
				delta_end.element(t-1,k) = -tempvv[k] ;
				path_ends.element(t-1,k) = (tempii[k]%N) ;
				ktable_ends.element(t-1,k) = (tempii[k]-(tempii[k]%N))/N ;
			}
		}
	}
	
	{ //state sequence backtracking
		max_best_iter=0 ;
		
		CArray<float64_t> sort_delta_end(max_iter*nbest) ;
		CArray<int16_t> sort_k(max_iter*nbest) ;
		CArray<int32_t> sort_t(max_iter*nbest) ;
		CArray<int32_t> sort_idx(max_iter*nbest) ;
		
		int32_t i=0 ;
		for (int32_t iter=0; iter<max_iter-1; iter++)
			for (int16_t k=0; k<nbest; k++)
			{
				sort_delta_end[i]=-delta_end.element(iter,k) ;
				sort_k[i]=k ;
				sort_t[i]=iter+1 ;
				sort_idx[i]=i ;
				i++ ;
			}
		
		CMath::qsort_index(sort_delta_end.get_array(), sort_idx.get_array(), (max_iter-1)*nbest) ;

		for (int16_t n=0; n<nbest; n++)
		{
			int16_t k=sort_k[sort_idx[n]] ;
			int32_t iter=sort_t[sort_idx[n]] ;
			prob_nbest[n]=-sort_delta_end[n] ;

			if (iter>max_best_iter)
				max_best_iter=iter ;
			
			ASSERT(k<nbest) ;
			ASSERT(iter<max_iter) ;
			
			paths.element(iter,n) = path_ends.element(iter-1, k) ;
			int16_t q   = ktable_ends.element(iter-1, k) ;
			
			for (int32_t t = iter; t>0; t--)
			{
				paths.element(t-1,n)=psi.element(t, paths.element(t,n), q);
				q = ktable.element(t, paths.element(t,n), q) ;
			}
		}
	}

	delete delta ;
	delete delta_new ;
}


void CDynProg::translate_from_single_order(
	uint16_t* obs, int32_t sequence_length, int32_t start, int32_t order,
	int32_t max_val)
{
	int32_t i,j;
	uint16_t value=0;
	
	for (i=sequence_length-1; i>=order-1; i--) //convert interval of size T
	{
		value=0;
		for (j=i; j>=i-order+1; j--)
			value= (value >> max_val) | (obs[j] << (max_val * (order-1)));
		
		obs[i]=value;
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
		//ASSERT(value<num_words) ;
	}
	if (start>0)
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
}

void CDynProg::reset_svm_value(
	int32_t pos, int32_t & last_svm_pos, float64_t * svm_value)
{
	for (int32_t i=0; i<num_words_single; i++)
		word_used_single[i]=false ;
	for (int32_t s=0; s<num_svms; s++)
		svm_value_unnormalized_single[s] = 0 ;
	for (int32_t s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
	last_svm_pos = pos - 6+1 ;
	num_unique_words_single=0 ;
}

void CDynProg::extend_svm_value(
	uint16_t* wordstr, int32_t pos, int32_t &last_svm_pos, float64_t* svm_value)
{
	bool did_something = false ;
	for (int32_t i=last_svm_pos-1; (i>=pos) && (i>=0); i--)
	{
		if (wordstr[i]>=num_words_single)
			SG_DEBUG( "wordstr[%i]=%i\n", i, wordstr[i]) ;
		
		if (!word_used_single[wordstr[i]])
		{
			for (int32_t s=0; s<num_svms_single; s++)
				svm_value_unnormalized_single[s]+=dict_weights.element(wordstr[i],s) ;
			
			word_used_single[wordstr[i]]=true ;
			num_unique_words_single++ ;
			did_something=true ;
		}
	} ;
	if (num_unique_words_single>0)
	{
		last_svm_pos=pos ;
		if (did_something)
			for (int32_t s=0; s<num_svms; s++)
				svm_value[s]= svm_value_unnormalized_single[s]/sqrt((float64_t)num_unique_words_single) ;  // full normalization
	}
	else
	{
		// what should I do?
		for (int32_t s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
}


void CDynProg::reset_segment_sum_value(
	int32_t num_states, int32_t pos, int32_t & last_segment_sum_pos,
	float64_t * segment_sum_value)
{
	for (int32_t s=0; s<num_states; s++)
		segment_sum_value[s] = 0 ;
	last_segment_sum_pos = pos ;
	//SG_DEBUG( "start: %i\n", pos) ;
}

void CDynProg::extend_segment_sum_value(
	float64_t *segment_sum_weights, int32_t seqlen, int32_t num_states,
	int32_t pos, int32_t &last_segment_sum_pos, float64_t* segment_sum_value)
{
	for (int32_t i=last_segment_sum_pos-1; (i>=pos) && (i>=0); i--)
	{
		for (int32_t s=0; s<num_states; s++)
			segment_sum_value[s] += segment_sum_weights[i*num_states+s] ;
	} ;
	//SG_DEBUG( "extend %i: %f\n", pos, segment_sum_value[0]) ;
	last_segment_sum_pos = pos ;
}


void CDynProg::best_path_2struct(
	const float64_t *seq_array, int32_t seq_len, const int32_t *pos,
	CPlifBase **Plif_matrix, const char *genestr, int32_t genestr_len,
	int16_t nbest, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, float64_t *dictionary_weights, int32_t dict_len,
	float64_t *segment_sum_weights)
{
	const int32_t default_look_back = 100 ;
	int32_t max_look_back = default_look_back ;
	bool use_svm = false ;
	ASSERT(dict_len==num_svms*num_words_single) ;
	dict_weights.set_array(dictionary_weights, dict_len, num_svms, false, false) ;
	dict_weights_array=dict_weights.get_array() ;

	CArray2<CPlifBase*> PEN(Plif_matrix, N, N, false) ;
	CArray2<float64_t> seq((float64_t *)seq_array, N, seq_len, false) ;
	
	float64_t svm_value[num_svms] ;
	float64_t segment_sum_value[N] ;
	
	{ // initialize svm_svalue
		for (int32_t s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
	{ // determine maximal length of look-back
		for (int32_t i=0; i<N; i++)
			for (int32_t j=0; j<N; j++)
			{
				CPlifBase *penij=PEN.element(i,j) ;
				if (penij==NULL)
					continue ;
				if (penij->get_max_value()>max_look_back)
					max_look_back=(int32_t) CMath::ceil(penij->get_max_value());
				if (penij->uses_svm_values())
					use_svm=true ;
			}
	}
	max_look_back = CMath::min(genestr_len, max_look_back) ;
	//SG_DEBUG("use_svm=%i\n", use_svm) ;
	//SG_DEBUG("max_look_back=%i\n", max_look_back) ;
	
	const int32_t look_back_buflen = (max_look_back+1)*nbest*N ;
	//SG_DEBUG("look_back_buflen=%i\n", look_back_buflen) ;
	const float64_t mem_use = (float64_t)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(int16_t)+sizeof(int32_t))+
								look_back_buflen*(2*sizeof(float64_t)+sizeof(int32_t))+
								seq_len*(sizeof(T_STATES)+sizeof(int32_t))+
								genestr_len*sizeof(bool))/(1024*1024)
		 ;
    bool is_big = (mem_use>200) || (seq_len>5000) ;
	
	if (is_big)
	{
		SG_DEBUG("calling best_path_2struct: seq_len=%i, N=%i, lookback=%i nbest=%i\n", 
					 seq_len, N, max_look_back, nbest) ;
		SG_DEBUG("allocating %1.2fMB of memory\n", 
					 mem_use) ;
	}
	ASSERT(nbest<32000) ;
		
	CArray3<float64_t> delta(max_look_back+1, N, nbest) ;
	CArray3<T_STATES> psi(seq_len,N,nbest) ;
	CArray3<int16_t> ktable(seq_len,N,nbest) ;
	CArray3<int32_t> ptable(seq_len,N,nbest) ;

	CArray<float64_t> delta_end(nbest) ;
	CArray<T_STATES> path_ends(nbest) ;
	CArray<int16_t> ktable_end(nbest) ;

	CArray<float64_t> tempvv(look_back_buflen) ;
	CArray<int32_t> tempii(look_back_buflen) ;

	CArray<T_STATES> state_seq(seq_len) ;
	CArray<int32_t> pos_seq(seq_len) ;

	// translate to words, if svm is used
	uint16_t* wordstr=NULL ;
	if (use_svm)
	{
		ASSERT(dictionary_weights!=NULL) ;
		wordstr=new uint16_t[genestr_len] ;
		for (int32_t i=0; i<genestr_len; i++)
			switch (genestr[i])
			{
			case 'A':
			case 'a': wordstr[i]=0 ; break ;
			case 'C':
			case 'c': wordstr[i]=1 ; break ;
			case 'G':
			case 'g': wordstr[i]=2 ; break ;
			case 'T':
			case 't': wordstr[i]=3 ; break ;
			default: ASSERT(0) ;
			}
		translate_from_single_order(wordstr, genestr_len, word_degree_single-1, word_degree_single) ;
	}
	
	
	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			delta.element(0,i,0) = get_p(i) + seq.element(i,0) ;
			psi.element(0,i,0)   = 0 ;
			ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (int16_t k=1; k<nbest; k++)
			{
				delta.element(0,i,k)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;
				ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (int32_t t=1; t<seq_len; t++)
	{
		if (is_big && t%(seq_len/10000)==1)
			SG_PROGRESS(t, 0, seq_len);
		//SG_PRINT("%i\n", t) ;
		
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (int16_t k=0; k<nbest; k++)
				{
					delta.element(t%max_look_back,j,k)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const float64_t *elem_val      = trans_list_forward_val[j] ;
				
				int32_t list_len=0 ;
				for (int32_t i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					//SG_DEBUG( "i=%i  ii=%i  num_elem=%i  PEN=%ld\n", i, ii, num_elem, PEN(j,ii)) ;
					
					const CPlifBase * penalty = PEN.element(j,ii) ;
					int32_t look_back = default_look_back ;
					if (penalty!=NULL)
						look_back=(int32_t) (CMath::ceil(penalty->get_max_value()));
					
					int32_t last_svm_pos ;
					if (use_svm)
						reset_svm_value(pos[t], last_svm_pos, svm_value) ;

					int32_t last_segment_sum_pos ;
					reset_segment_sum_value(N, pos[t], last_segment_sum_pos, segment_sum_value) ;

					for (int32_t ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						if (use_svm)
							extend_svm_value(wordstr, pos[ts], last_svm_pos, svm_value) ;

						extend_segment_sum_value(segment_sum_weights, seq_len, N, pos[ts], last_segment_sum_pos, segment_sum_value) ;
						
						float64_t pen_val = 0.0 ;
						if (penalty)
							pen_val=penalty->lookup_penalty(pos[t]-pos[ts], svm_value) + segment_sum_value[j] ;
						for (int16_t diff=0; diff<nbest; diff++)
						{
							float64_t  val        = delta.element(ts%max_look_back,ii,diff) + elem_val[i] ;
							val             += pen_val ;
							
							tempvv[list_len] = -val ;
							tempii[list_len] =  ii + diff*N + ts*N*nbest;
							//SG_DEBUG( "%i (%i,%i,%i, %i, %i) ", list_len, diff, ts, i, pos[t]-pos[ts], look_back) ;
							list_len++ ;
						}
					}
				}
				CMath::nmin<int32_t>(tempvv.get_array(), tempii.get_array(), list_len, nbest) ;
				
				for (int16_t k=0; k<nbest; k++)
				{
					if (k<list_len)
					{
						delta.element(t%max_look_back,j,k)    = -tempvv[k] + seq.element(j,t);
						psi.element(t,j,k)      = (tempii[k]%N) ;
						ktable.element(t,j,k)     = (tempii[k]%(N*nbest)-psi.element(t,j,k))/N ;
						ptable.element(t,j,k)     = (tempii[k]-(tempii[k]%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(t%max_look_back,j,k)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
			}
		}
	}
	
	{ //termination
		int32_t list_len = 0 ;
		for (int16_t diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				tempvv[list_len] = -(delta.element((seq_len-1)%max_look_back,i,diff)+get_q(i)) ;
				tempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(tempvv.get_array(), tempii.get_array(), list_len, nbest) ;
		
		for (int16_t k=0; k<nbest; k++)
		{
			delta_end.element(k) = -tempvv[k] ;
			path_ends.element(k) = (tempii[k]%N) ;
			ktable_end.element(k) = (tempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (int16_t k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			int32_t i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			int16_t q   = ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			int32_t num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*(seq_len+1)] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*(seq_len+1)]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*(seq_len+1)]=-1 ;
			my_pos_seq[num_states+k*(seq_len+1)]=-1 ;
		}
	}
	if (is_big)
		SG_PRINT( "DONE.     \n") ;
}

/*void CDynProg::reset_svm_values(int32_t pos, int32_t * last_svm_pos, float64_t * svm_value) 
{
	for (int32_t j=0; j<num_degrees; j++)
	{
		for (int32_t i=0; i<num_words_array[j]; i++)
			word_used.element(word_used_array, j, i, num_degrees)=false ;
		for (int32_t s=0; s<num_svms; s++)
			svm_values_unnormalized.element(j,s) = 0 ;
		num_unique_words[j]=0 ;
		last_svm_pos[j] = pos - word_degree[j]+1 ;
		svm_pos_start[j] = pos - word_degree[j] ;
	}
	for (int32_t s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
}

void CDynProg::extend_svm_values(uint16_t** wordstr, int32_t pos, int32_t *last_svm_pos, float64_t* svm_value) 
{
	bool did_something = false ;
	for (int32_t j=0; j<num_degrees; j++)
	{
		for (int32_t i=last_svm_pos[j]-1; (i>=pos) && (i>=0); i--)
		{
			if (wordstr[j][i]>=num_words_array[j])
				SG_DEBUG( "wordstr[%i]=%i\n", i, wordstr[j][i]) ;

			ASSERT(wordstr[j][i]<num_words_array[j]) ;
			if (!word_used.element(word_used_array, j, wordstr[j][i], num_degrees))
			{
				for (int32_t s=0; s<num_svms; s++)
					svm_values_unnormalized.element(j,s)+=dict_weights_array[wordstr[j][i]+cum_num_words_array[j]+s*cum_num_words_array[num_degrees]] ;
					//svm_values_unnormalized.element(j,s)+=dict_weights.element(wordstr[j][i]+cum_num_words_array[j],s) ;
				
				//word_used.element(j,wordstr[j][i])=true ;
				word_used.element(word_used_array, j, wordstr[j][i], num_degrees)=true ;
				num_unique_words[j]++ ;
				did_something=true ;
			} ;
		} ;
		if (num_unique_words[j]>0)
			last_svm_pos[j]=pos ;
	} ;
	
	if (did_something)
		for (int32_t s=0; s<num_svms; s++)
		{
			svm_value[s]=0.0 ;
			for (int32_t j=0; j<num_degrees; j++)
				if (num_unique_words[j]>0)
					svm_value[s]+= svm_values_unnormalized.element(j,s)/sqrt((float64_t)num_unique_words[j]) ;  // full normalization
		}
}
*/

void CDynProg::init_segment_loss(
	struct segment_loss_struct & loss, int32_t seqlen, int32_t howmuchlookback)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	int32_t clear_size = CMath::min(howmuchlookback,seqlen) ;
	
	if (!loss.num_segment_id)
	{
		loss.segments_changed       = new int32_t[seqlen] ;
		loss.num_segment_id         = new float64_t[(max_a_id+1)*seqlen] ;
		loss.length_segment_id      = new int32_t[(max_a_id+1)*seqlen] ;

		clear_size = seqlen ;
	}
	
	for (int32_t j=0; j<clear_size; j++)
	{
		loss.segments_changed[j]=0 ;
		for (int32_t i=0; i<max_a_id+1; i++)       
		{
			loss.num_segment_id[i*seqlen+j] = 0;
			loss.length_segment_id[i*seqlen+j] = 0;
		}
	}

	loss.maxlookback = howmuchlookback ;
	loss.seqlen = seqlen;

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	segment_init_time += MyTime.time_diff_sec() ;
#endif
}

void CDynProg::clear_segment_loss(struct segment_loss_struct & loss) 
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	
	if (loss.num_segment_id != NULL)
	{
		delete[] loss.segments_changed ;
		delete[] loss.num_segment_id ;
		delete[] loss.length_segment_id ;
		loss.segments_changed = NULL ;
		loss.num_segment_id = NULL ;
		loss.length_segment_id = NULL ;
	}
#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	segment_clean_time += MyTime.time_diff_sec() ;
#endif
}

float64_t CDynProg::extend_segment_loss(
	struct segment_loss_struct & loss, const int32_t * pos_array,
	int32_t segment_id, int32_t pos, int32_t & last_pos, float64_t &last_value)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	
	if (pos==last_pos)
		return last_value ;
	ASSERT(pos<last_pos) ;

	last_pos-- ;
	bool changed = false ;
	while (last_pos>=pos)
	{
		if (loss.segments_changed[last_pos])
		{
			changed=true ;
			break ;
		}
		last_pos-- ;
	}
	if (last_pos<pos)
		last_pos = pos ;
	
	if (!changed)
	{
		ASSERT(last_pos>=0) ;
		ASSERT(last_pos<loss.seqlen) ;
		float64_t length_contrib = (pos_array[last_pos]-pos_array[pos])*m_segment_loss.element(m_segment_ids.element(pos), segment_id, 1) ;
		float64_t ret = last_value + length_contrib ;
		last_pos = pos ;
		return ret ;
	}

	CArray2<float64_t> num_segment_id(loss.num_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	CArray2<int32_t> length_segment_id(loss.length_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	float64_t ret = 0.0 ;
	for (int32_t i=0; i<max_a_id+1; i++)
	{
		//SG_DEBUG( "%i: %i, %i, %f (%f), %f (%f)\n", pos, num_segment_id.element(pos, i), length_segment_id.element(pos, i), num_segment_id.element(pos, i)*m_segment_loss.element(i, segment_id,0), m_segment_loss.element(i, segment_id, 0), length_segment_id.element(pos, i)*m_segment_loss.element(i, segment_id, 1), m_segment_loss.element(i, segment_id,1)) ;

		if (num_segment_id.element(pos, i)!=0)
		{
			ret += num_segment_id.element(pos, i)*m_segment_loss.element(i, segment_id, 0) ;
		//	SG_PRINT("ret:%f pos:%i i:%i segment_id:%i \n",ret,pos,i,segment_id);
		//	if (ret>0)
		//	{
		//		for (int32_t g=0; g<max_a_id+1; g++)
		//			SG_PRINT("g:%i sid(pos, g):%i    ",g,num_segment_id.element(pos, g));
		//		SG_PRINT("\n");
		//	}
		}
		if (length_segment_id.element(pos, i)!=0)
			ret += length_segment_id.element(pos, i)*m_segment_loss.element(i, segment_id, 1) ;
	}
	last_pos = pos ;
	last_value = ret ;

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	segment_extend_time += MyTime.time_diff_sec() ;
#endif
	return ret ;
}

void CDynProg::find_segment_loss_till_pos(
	const int32_t * pos, int32_t t_end, CArray<int32_t>& segment_ids,
	CArray<float64_t>& segment_mask, struct segment_loss_struct & loss)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	CArray2<float64_t> num_segment_id(loss.num_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	CArray2<int32_t> length_segment_id(loss.length_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	
	for (int32_t i=0; i<max_a_id+1; i++)
	{
		num_segment_id.element(t_end, i) = 0 ;
		length_segment_id.element(t_end, i) = 0 ;
	}
	int32_t wobble_pos_segment_id_switch = 0 ;
	int32_t last_segment_id = -1 ;
	int32_t ts = t_end-1 ;       
	while ((ts>=0) && (pos[t_end] - pos[ts] <= loss.maxlookback))
	{
		int32_t cur_segment_id = segment_ids.element(ts) ;
		// allow at most one wobble
		bool wobble_pos = (CMath::abs(segment_mask.element(ts))<1e-7) && (wobble_pos_segment_id_switch==0) ;
		if (!(cur_segment_id<=max_a_id))
			SG_ERROR("(cur_segment_id<=max_a_id), cur_segment_id:%i max_a_id:%i\n",cur_segment_id,max_a_id);
		ASSERT(cur_segment_id>=0) ;
		
		for (int32_t i=0; i<max_a_id+1; i++)
		{
			num_segment_id.element(ts, i) = num_segment_id.element(ts+1, i) ;
			length_segment_id.element(ts, i) = length_segment_id.element(ts+1, i) ;
		}
		
		if (cur_segment_id!=last_segment_id)
		{
			if (wobble_pos)
			{
				//SG_DEBUG( "no change at %i: %i, %i\n", ts, last_segment_id, cur_segment_id) ;
				wobble_pos_segment_id_switch++ ;
				//ASSERT(wobble_pos_segment_id_switch<=1) ;
			}
			else
			{
				//SG_DEBUG( "change at %i: %i, %i\n", ts, last_segment_id, cur_segment_id) ;
				loss.segments_changed[ts] = true ;
				num_segment_id.element(ts, cur_segment_id) += segment_mask.element(ts);
				length_segment_id.element(ts, cur_segment_id) += (int32_t)((pos[ts+1]-pos[ts])*segment_mask.element(ts));
				wobble_pos_segment_id_switch = 0 ;
			}
			last_segment_id = cur_segment_id ;
		} 
		else
			if (!wobble_pos)
				length_segment_id.element(ts, cur_segment_id) += pos[ts+1] - pos[ts] ;

		ts--;
	}
#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	segment_pos_time += MyTime.time_diff_sec() ;
#endif
}

void CDynProg::init_svm_values(
	struct svm_values_struct & svs, int32_t start_pos, int32_t seqlen,
	int32_t maxlookback)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	/*
	  See find_svm_values_till_pos for comments
	  
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  where t_end is the end of all segments we are currently looking at
	*/
	int32_t clear_size = CMath::min(maxlookback,seqlen) ;
	
	if (!svs.svm_values)
	{
		svs.svm_values              = new float64_t[seqlen*num_svms] ;
		svs.num_unique_words        = new int32_t*[num_degrees] ;
		svs.svm_values_unnormalized = new float64_t*[num_degrees] ;
		svs.word_used               = new bool**[num_degrees] ;
		for (int32_t j=0; j<num_degrees; j++)
		{
			svs.word_used[j]               = new bool*[num_svms] ;
			for (int32_t s=0; s<num_svms; s++)
				svs.word_used[j][s]               = new bool[num_words_array[j]] ;
		}
		for (int32_t j=0; j<num_degrees; j++)
		{
			svs.svm_values_unnormalized[j] = new float64_t[num_svms] ;
			svs.num_unique_words[j]        = new int32_t[num_svms] ;
		}
		svs.start_pos               = new int32_t[num_svms] ;
		clear_size = seqlen ;
	}
	
	//for (int32_t i=0; i<maxlookback*num_svms; i++)       // initializing this for safety, though we should be able to live without it
	//	svs.svm_values[i] = 0;
	memset(svs.svm_values, 0, clear_size*num_svms*sizeof(float64_t)) ;

	for (int32_t j=0; j<num_degrees; j++)
	{		
		//for (int32_t s=0; s<num_svms; s++)
		//	svs.svm_values_unnormalized[j][s] = 0 ;
		memset(svs.svm_values_unnormalized[j], 0, num_svms*sizeof(float64_t)) ;

		//for (int32_t s=0; s<num_svms; s++)
		//  svs.num_unique_words[j][s] = 0 ;
		memset(svs.num_unique_words[j], 0, num_svms*sizeof(int32_t)) ;
	}
	
	for (int32_t j=0; j<num_degrees; j++)
		for (int32_t s=0; s<num_svms; s++)
		{
			//for (int32_t i=0; i<num_words_array[j]; i++)
			//	svs.word_used[j][s][i] = false ;
			memset(svs.word_used[j][s], 0, num_words_array[j]*sizeof(bool)) ;
		}
	
	for (int32_t s=0; s<num_svms; s++)
		svs.start_pos[s] = start_pos - mod_words.element(s,1) ;
	
	svs.maxlookback = maxlookback ;
	svs.seqlen = seqlen ;

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	svm_init_time += MyTime.time_diff_sec() ;
#endif
}

void CDynProg::clear_svm_values(struct svm_values_struct & svs) 
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif	
	if (NULL != svs.svm_values)
	{
		for (int32_t j=0; j<num_degrees; j++)
		{
			for (int32_t s=0; s<num_svms; s++)
				delete[] svs.word_used[j][s] ;
			delete[] svs.word_used[j];
		}
		delete[] svs.word_used;
		
		for (int32_t j=0; j<num_degrees; j++)
			delete[] svs.svm_values_unnormalized[j] ;
		for (int32_t j=0; j<num_degrees; j++)
			delete[] svs.num_unique_words[j] ;
		
		delete[] svs.svm_values_unnormalized;
		delete[] svs.svm_values;
		delete[] svs.num_unique_words ;
		
		svs.word_used=NULL ;
		svs.svm_values=NULL ;
		svs.svm_values_unnormalized=NULL ;
	}

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	svm_clean_time += MyTime.time_diff_sec() ;
#endif
}


void CDynProg::find_svm_values_till_pos(
	uint16_t*** wordstr,  const int32_t *pos,  int32_t t_end,
	struct svm_values_struct &svs)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	
	/*
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095 
	  corresponding to the 6-mer in genestr(i-5:i) 
	  pos is a vector of candidate transition positions (it is input to best_path_trans)
	  t_end is some index in pos
	  
	  svs has been initialized by init_svm_values
	  
	  At the end of this procedure, 
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  The SVM weights are precomputed in dict_weights
	*/
	
	for (int32_t j=0; j<num_degrees; j++)
	//for (int32_t j=0; j<1; j++)
	{
		int32_t plen = 1;
		int32_t ts = t_end-1;        // index in pos; pos(ts) and pos(t) are indices of wordstr
		int32_t offset;
		
		int32_t posprev = pos[t_end]-word_degree[j]+1;
		int32_t poscurrent = pos[ts];
		
		//SG_DEBUG( "j=%i seqlen=%i posprev = %i, poscurrent = %i", j, svs.seqlen, posprev, poscurrent) ;

		if (poscurrent<0)
			poscurrent = 0;
		float64_t * my_svm_values_unnormalized = svs.svm_values_unnormalized[j] ;
		int32_t * my_num_unique_words = svs.num_unique_words[j] ;
		bool ** my_word_used = svs.word_used[j] ;
		
		int32_t len = pos[t_end] - poscurrent;
		while ((ts>=0) && (len <= svs.maxlookback))
		{
			for (int32_t i=posprev-1 ; (i>=poscurrent) && (i>=0) ; i--)
			{
				//SG_PRINT("string_words_array[0]=%i (%ld), j=%i (%ld)  i=%i\n", string_words_array[0], wordstr[string_words_array[0]], j, wordstr[string_words_array[0]][j], i) ;
				
				uint16_t word = wordstr[string_words_array[0]][j][i] ;
				int32_t last_string = string_words_array[0] ;
				for (int32_t s=0; s<num_svms; s++)
				{
				//sign_words_array[s]=false;
					// try to avoid memory accesses
					if (last_string != string_words_array[s])
					{
						last_string = string_words_array[s] ;
						word = wordstr[last_string][j][i] ;
					}

					// do not consider k-mer, if seen before and in signum mode
					if (sign_words_array[s] && my_word_used[s][word])
						continue ;
					
					// only count k-mer if in frame (if applicable)
					//if ((svs.start_pos[s]-i>0) && ((svs.start_pos[s]-i)%mod_words_array[s]==0))
					{
						my_svm_values_unnormalized[s] += dict_weights_array[(word+cum_num_words_array[j])+s*cum_num_words_array[num_degrees]] ;
						//svs.svm_values_unnormalized[j][s]+=dict_weights.element(word+cum_num_words_array[j], s) ;
						my_num_unique_words[s]++ ;
						if (sign_words_array[s])
							my_word_used[s][word]=true ;
					}
				}
			}
			offset = plen*num_svms ;
			for (int32_t s=0; s<num_svms; s++)
			{
				float64_t normalization_factor = 1.0;
				if (my_num_unique_words[s] > 0)
				{
					if (sign_words_array[s])
						normalization_factor = sqrt((float64_t)my_num_unique_words[s]);
					else
						normalization_factor = (float64_t)my_num_unique_words[s];
				}

				if (j==0)
					svs.svm_values[offset+s]=0 ;
				svs.svm_values[offset+s] += my_svm_values_unnormalized[s] / normalization_factor;
			}
			
			if (posprev > poscurrent)         // remember posprev initially set to pos[t_end]-word_degree+1... pos[ts] could be e.g. pos[t_end]-2
				posprev = poscurrent;           
			
			ts--;
			plen++;
			
			if (ts>=0)
			{
				poscurrent=pos[ts];
				if (poscurrent<0)
					poscurrent = 0;
				len = pos[t_end] - poscurrent;
			}
		}
	}

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	svm_pos_time += MyTime.time_diff_sec() ;
#endif
}


void CDynProg::find_svm_values_till_pos(
	uint16_t** wordstr,  const int32_t *pos,  int32_t t_end,
	struct svm_values_struct &svs)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif	
	/*
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095 
	  corresponding to the 6-mer in genestr(i-5:i) 
	  pos is a vector of candidate transition positions (it is input to best_path_trans)
	  t_end is some index in pos
	  
	  svs has been initialized by init_svm_values
	  
	  At the end of this procedure, 
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  The SVM weights are precomputed in dict_weights
	*/
	
	for (int32_t j=0; j<num_degrees; j++)
	//for (int32_t j=0; j<1; j++)
	{
		int32_t plen = 1;
		int32_t ts = t_end-1;        // index in pos; pos(ts) and pos(t) are indices of wordstr
		int32_t offset;
		
		int32_t posprev = pos[t_end]-word_degree[j]+1;
		int32_t poscurrent = pos[ts];
		
		//SG_DEBUG( "j=%i seqlen=%i posprev = %i, poscurrent = %i", j, svs.seqlen, posprev, poscurrent) ;

		if (poscurrent<0)
			poscurrent = 0;
		float64_t * my_svm_values_unnormalized = svs.svm_values_unnormalized[j] ;
		int32_t * my_num_unique_words = svs.num_unique_words[j] ;
		bool ** my_word_used = svs.word_used[j] ;
		
		int32_t len = pos[t_end] - poscurrent;
		while ((ts>=0) && (len <= svs.maxlookback))
		{
			for (int32_t i=posprev-1 ; (i>=poscurrent) && (i>=0) ; i--)
			{
				//SG_PRINT("string_words_array[0]=%i (%ld), j=%i (%ld)  i=%i\n", string_words_array[0], wordstr[string_words_array[0]], j, wordstr[string_words_array[0]][j], i) ;
				
				uint16_t word = wordstr[j][i] ;
				for (int32_t s=0; s<num_svms; s++)
				{
					//sign_words_array[s]=false;
					// do not consider k-mer, if seen before and in signum mode
					if (sign_words_array[s] && my_word_used[s][word])
						continue ;
					
					// only count k-mer if in frame (if applicable)
					//if ((svs.start_pos[s]-i>0) && ((svs.start_pos[s]-i)%mod_words_array[s]==0))
					{
						my_svm_values_unnormalized[s] += dict_weights_array[(word+cum_num_words_array[j])+s*cum_num_words_array[num_degrees]] ;
						//svs.svm_values_unnormalized[j][s]+=dict_weights.element(word+cum_num_words_array[j], s) ;
						my_num_unique_words[s]++ ;
						if (sign_words_array[s])
							my_word_used[s][word]=true ;
					}
				}
			}
			offset = plen*num_svms ;
			for (int32_t s=0; s<num_svms; s++)
			{
				float64_t normalization_factor = 1.0;
				if (my_num_unique_words[s] > 0)
				{
					if (sign_words_array[s])
						normalization_factor = sqrt((float64_t)my_num_unique_words[s]);
					else
						normalization_factor = (float64_t)my_num_unique_words[s];
				}

				if (j==0)
					svs.svm_values[offset+s]=0 ;
				svs.svm_values[offset+s] += my_svm_values_unnormalized[s] / normalization_factor;
			}
			
			if (posprev > poscurrent)         // remember posprev initially set to pos[t_end]-word_degree+1... pos[ts] could be e.g. pos[t_end]-2
				posprev = poscurrent;           
			
			ts--;
			plen++;
			
			if (ts>=0)
			{
				poscurrent=pos[ts];
				if (poscurrent<0)
					poscurrent = 0;
				len = pos[t_end] - poscurrent;
			}
		}
	}

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	svm_pos_time += MyTime.time_diff_sec() ;
#endif
}

bool CDynProg::extend_orf(
	int32_t orf_from, int32_t orf_to, int32_t start, int32_t &last_pos,
	int32_t to)
{
#ifdef DYNPROG_TIMING
	MyTime.start() ;
#endif
	
	if (start<0) 
		start=0 ;
	if (to<0)
		to=0 ;
	
	int32_t orf_target = orf_to-orf_from ;
	if (orf_target<0) orf_target+=3 ;
	
	int32_t pos ;
	if (last_pos==to)
		pos = to-orf_to-3 ;
	else
		pos=last_pos ;

	if (pos<0)
		return true ;
	
	for (; pos>=start; pos-=3)
		if (m_genestr_stop[pos])
			return false ;
	
	last_pos = CMath::min(pos+3,to-orf_to-3) ;

#ifdef DYNPROG_TIMING
	MyTime.stop() ;
	orf_time += MyTime.time_diff_sec() ;
#endif
	return true ;
}

template <int16_t nbest, bool with_loss, bool with_multiple_sequences>
void CDynProg::best_path_trans(
	const float64_t *seq_array, int32_t seq_len, const int32_t *pos,
	const int32_t *orf_info_array, CPlifBase **Plif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,
	int32_t genestr_num, float64_t *prob_nbest, int32_t *my_state_seq,
	int32_t *my_pos_seq, bool use_orf)
{
#ifdef DYNPROG_TIMING
	segment_init_time = 0.0 ;
	segment_pos_time = 0.0 ;
	segment_extend_time = 0.0 ;
	segment_clean_time = 0.0 ;
	orf_time = 0.0 ;
	svm_init_time = 0.0 ;
	svm_pos_time = 0.0 ;
	svm_clean_time = 0.0 ;
	MyTime2.start() ;
#endif
	
	//SG_PRINT( "best_path_trans:%x\n", seq_array);
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	}
	
#ifdef DYNPROG_DEBUG
	transition_matrix_a.set_name("transition_matrix");
	transition_matrix_a.display_array();
	mod_words.display_array() ;
	sign_words.display_array() ;
	string_words.display_array() ;
	SG_PRINT("use_orf = %i\n", use_orf) ;
#endif
	
	int32_t max_look_back = 20000 ;
	bool use_svm = false ;
	//ASSERT(dict_len==num_svms*cum_num_words_array[num_degrees]) ;
	//dict_weights.set_array(dictionary_weights, cum_num_words_array[num_degrees], num_svms, false, false) ;
	//dict_weights_array=dict_weights.get_array() ;
	
	SG_PRINT("N:%i, seq_len:%i, max_num_signals:%i\n",N, seq_len, max_num_signals) ;

//	for (int32_t i=0;i<N*seq_len*max_num_signals;i++)
//		SG_PRINT("(%i)%0.2f ",i,seq_array[i]);

	//CArray2<CPlifBase*> PEN(Plif_matrix, N, N, false, false) ;
	CArray2<CPlifBase*> PEN(Plif_matrix, N, N, false, true) ;
	PEN.set_name("PEN");
	//CArray2<CPlifBase*> PEN_state_signals(Plif_state_signals, N, max_num_signals, false, false) ;
	CArray2<CPlifBase*> PEN_state_signals(Plif_state_signals, N, max_num_signals, false, true) ;
	PEN_state_signals.set_name("state_signals");
	//CArray3<float64_t> seq_input(seq_array, N, seq_len, max_num_signals) ;
	CArray3<float64_t> seq_input(seq_array, N, seq_len, max_num_signals) ;
	seq_input.set_name("seq_input") ;
	//seq_input.display_array() ;
	CArray2<float64_t> seq(N, seq_len) ;
	seq.set_name("seq") ;
	seq.zero() ;

	CArray2<int32_t> orf_info(orf_info_array, N, 2) ;
	orf_info.set_name("orf_info") ;
	//g_orf_info = orf_info ;
	//orf_info.display_array() ;

	float64_t svm_value[m_num_lin_feat_plifs_cum[m_num_raw_data]] ;
	{ // initialize svm_svalue
		for (int32_t s=0; s<m_num_lin_feat_plifs_cum[m_num_raw_data]; s++)
			svm_value[s]=0 ;
	}

	{ // convert seq_input to seq
      // this is independent of the svm values 
		for (int32_t i=0; i<N; i++)
			for (int32_t j=0; j<seq_len; j++)
				seq.element(i,j) = 0 ;

		for (int32_t i=0; i<N; i++)
			for (int32_t j=0; j<seq_len; j++)
				for (int32_t k=0; k<max_num_signals; k++)
				{
					if ((PEN_state_signals.element(i,k)==NULL) && (k==0))
					{
						// no plif
						seq.element(i,j) = seq_input.element(i,j,k) ;
						break ;
					}
					if (PEN_state_signals.element(i,k)!=NULL)
					{
						// just one plif
						if (CMath::finite(seq_input.element(i,j,k)))
							seq.element(i,j) += PEN_state_signals.element(i,k)->lookup_penalty(seq_input.element(i,j,k), svm_value) ;
						else
							// keep infinity values
							seq.element(i,j) = seq_input.element(i, j, k) ;
					} 
					else
						break ;
				}
	}


	{ // determine maximal length of look-back
		for (int32_t i=0; i<N; i++)
		{
			// only consider transitions that are actually allowed
			const T_STATES num_elem   = trans_list_forward_cnt[i] ;
			const T_STATES *elem_list = trans_list_forward[i] ;
				
			for (int32_t jj=0; jj<num_elem; jj++)
			{
				T_STATES j = elem_list[jj] ;

				CPlifBase *penij=PEN.element(i,j) ;
				if (penij==NULL)
					continue ;
				if (penij->get_max_value()>max_look_back)
				{
					SG_DEBUG( "%d %d -> value: %f\n", i,j,penij->get_max_value());
					//penij->print() ;
					max_look_back=(int32_t) (CMath::ceil(penij->get_max_value()));
				}
				if (penij->uses_svm_values())
					use_svm=true ;
			}
		}
	}

	//SG_PRINT("use_svm=%i, genestr_len: \n", use_svm, m_genestr_len) ;
	max_look_back = CMath::min(m_genestr_len, max_look_back) ;
	SG_DEBUG("use_svm=%i\n", use_svm) ;
	
	SG_DEBUG("maxlook: %d N: %d nbest: %d \n", max_look_back, N, nbest);
	const int32_t look_back_buflen = (max_look_back*N+1)*nbest ;
	SG_DEBUG("look_back_buflen=%i\n", look_back_buflen) ;
	const float64_t mem_use = (float64_t)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(int16_t)+sizeof(int32_t))+
								  look_back_buflen*(2*sizeof(float64_t)+sizeof(int32_t))+
								  seq_len*(sizeof(T_STATES)+sizeof(int32_t))+
								  m_genestr_len*sizeof(bool))/(1024*1024);

    bool is_big = (mem_use>200) || (seq_len>5000) ;

	if (1)//(is_big)
	{
		SG_DEBUG("calling best_path_trans: seq_len=%i, N=%i, lookback=%i nbest=%i\n", 
					 seq_len, N, max_look_back, nbest) ;
		SG_DEBUG("allocating %1.2fMB of memory\n", 
					 mem_use) ;
	}
	ASSERT(nbest<32000) ;
	

	
	CArray3<float64_t> delta(seq_len, N, nbest) ;
	delta.set_name("delta");
	float64_t* delta_array = delta.get_array() ;
	//delta.zero() ;
	
	CArray3<T_STATES> psi(seq_len, N, nbest) ;
	psi.set_name("psi");
	//psi.zero() ;
	
	CArray3<int16_t> ktable(seq_len, N, nbest) ;
	ktable.set_name("ktable");
	//ktable.zero() ;
	
	CArray3<int32_t> ptable(seq_len, N, nbest) ;	
	ptable.set_name("ptable");
	//ptable.zero() ;

	CArray<float64_t> delta_end(nbest) ;
	delta_end.set_name("delta_end");
	//delta_end.zero() ;
	
	CArray<T_STATES> path_ends(nbest) ;
	path_ends.set_name("path_ends");
	//path_ends.zero() ;
	
	CArray<int16_t> ktable_end(nbest) ;
	ktable_end.set_name("ktable_end");
	//ktable_end.zero() ;

	float64_t * fixedtempvv=new float64_t[look_back_buflen] ;
	memset(fixedtempvv, 0, look_back_buflen*sizeof(float64_t)) ;
	int32_t * fixedtempii=new int32_t[look_back_buflen] ;
	memset(fixedtempii, 0, look_back_buflen*sizeof(int32_t)) ;

	CArray<float64_t> oldtempvv(look_back_buflen) ;
	oldtempvv.set_name("oldtempvv");
	CArray<float64_t> oldtempvv2(look_back_buflen) ;
	oldtempvv2.set_name("oldtempvv2");
	//oldtempvv.zero() ;
	//oldtempvv.display_size() ;
	
	CArray<int32_t> oldtempii(look_back_buflen) ;
	oldtempii.set_name("oldtempii");
	CArray<int32_t> oldtempii2(look_back_buflen) ;
	oldtempii2.set_name("oldtempii2");
	//oldtempii.zero() ;

	CArray<T_STATES> state_seq(seq_len) ;
	state_seq.set_name("state_seq");
	//state_seq.zero() ;
	
	CArray<int32_t> pos_seq(seq_len) ;
	pos_seq.set_name("pos_seq");
	//pos_seq.zero() ;

	
	//dict_weights.set_name("dict_weights") ;
	word_degree.set_name("word_degree") ;
	cum_num_words.set_name("cum_num_words") ;
	num_words.set_name("num_words") ;
	//word_used.set_name("word_used") ;
	//svm_values_unnormalized.set_name("svm_values_unnormalized") ;
	svm_pos_start.set_name("svm_pos_start") ;
	num_unique_words.set_name("num_unique_words") ;

	PEN.set_name("PEN") ;
	seq.set_name("seq") ;
	orf_info.set_name("orf_info") ;
	
	delta.set_name("delta") ;
	psi.set_name("psi") ;
	ktable.set_name("ktable") ;
	ptable.set_name("ptable") ;
	delta_end.set_name("delta_end") ;
	path_ends.set_name("path_ends") ;
	ktable_end.set_name("ktable_end") ;

#ifdef USE_TMP_ARRAYCLASS
	fixedtempvv.set_name("fixedtempvv") ;
	fixedtempii.set_name("fixedtempvv") ;
#endif

	oldtempvv.set_name("oldtempvv") ;
	oldtempvv2.set_name("oldtempvv2") ;
	oldtempii.set_name("oldtempii") ;
	oldtempii2.set_name("oldtempii2") ;


	//////////////////////////////////////////////////////////////////////////////// 

#ifdef DYNPROG_DEBUG
	state_seq.display_size() ;
	pos_seq.display_size() ;

	dict_weights.display_size() ;
	word_degree.display_array() ;
	cum_num_words.display_array() ;
	num_words.display_array() ;
	//word_used.display_size() ;
	//svm_values_unnormalized.display_size() ;
	svm_pos_start.display_array() ;
	num_unique_words.display_array() ;

	PEN.display_size() ;
	PEN_state_signals.display_size() ;
	seq.display_size() ;
	orf_info.display_size() ;
	
	//m_genestr_stop.display_size() ;
	delta.display_size() ;
	psi.display_size() ;
	ktable.display_size() ;
	ptable.display_size() ;
	delta_end.display_size() ;
	path_ends.display_size() ;
	ktable_end.display_size() ;

#ifdef USE_TMP_ARRAYCLASS
	fixedtempvv.display_size() ;
	fixedtempii.display_size() ;
#endif

	//oldtempvv.display_size() ;
    //oldtempii.display_size() ;

	state_seq.display_size() ;
	pos_seq.display_size() ;

	CArray<int32_t>pp = CArray<int32_t>(pos, seq_len) ;
	pp.set_name("pp");
	pp.display_array() ;
	
	//seq.zero() ;
	//seq_input.display_array() ;

#endif //DYNPROG_DEBUG

////////////////////////////////////////////////////////////////////////////////



	{
		for (int32_t s=0; s<num_svms; s++)
			ASSERT(string_words_array[s]<genestr_num)  ;
	}

	
    	//CArray2<int32_t*> trans_matrix_svms(N,N);
  	//CArray2<int32_t> trans_matrix_num_svms(N,N);

	{ // initialization

		for (T_STATES i=0; i<N; i++)
		{
			//delta.element(0, i, 0) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			delta.element(delta_array, 0, i, 0, seq_len, N) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			psi.element(0,i,0)   = 0 ;
			if (nbest>1)
				ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (int16_t k=1; k<nbest; k++)
			{
				int32_t dim1, dim2, dim3 ;
				delta.get_array_size(dim1, dim2, dim3) ;
				//SG_DEBUG("i=%i, k=%i -- %i, %i, %i\n", i, k, dim1, dim2, dim3) ;
				//delta.element(0, i, k)    = -CMath::INFTY ;
				delta.element(delta_array, 0, i, k, seq_len, N)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;                  // <--- what's this for?
				if (nbest>1)
					ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
/*
			for (T_STATES j=0; j<N; j++)
			{
				CPlifBase * penalty = PEN.element(i,j) ;
				int32_t num_current_svms=0;
				int32_t svm_ids[] = {-8, -7, -6, -5, -4, -3, -2, -1};
				if (penalty)
				{
					SG_PRINT("trans %i -> %i \n",i,j);
					penalty->get_used_svms(&num_current_svms, svm_ids);
					trans_matrix_svms.set_element(svm_ids,i,j);
					for (int32_t l=0;l<num_current_svms;l++)
						SG_PRINT("svm_ids[%i]: %i \n",l,svm_ids[l]);
					trans_matrix_num_svms.set_element(num_current_svms,i,j);
				}
			}
*/

		}
	}

	/*struct svm_values_struct svs;
	svs.num_unique_words = NULL;
	svs.svm_values = NULL;
	svs.svm_values_unnormalized = NULL;
	svs.word_used = NULL;*/

	struct segment_loss_struct loss;
	loss.segments_changed = NULL;
	loss.num_segment_id = NULL;

	SG_DEBUG("START_RECURSION \n\n");

	// recursion
	for (int32_t t=1; t<seq_len; t++)
	{
		if (is_big && t%(1+(seq_len/1000))==1)
			SG_PROGRESS(t, 0, seq_len);
		//SG_PRINT("%i\n", t) ;

		if (with_loss)
		{
			init_segment_loss(loss, seq_len, max_look_back);
			find_segment_loss_till_pos(pos, t, m_segment_ids, m_segment_mask, loss);  
		}
		
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<=-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (int16_t k=0; k<nbest; k++)
				{
					delta.element(delta_array, t, j, k, seq_len, N)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					if (nbest>1)
						ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const float64_t *elem_val      = trans_list_forward_val[j] ;
				const int32_t *elem_id      = trans_list_forward_id[j] ;
				
				int32_t fixed_list_len = 0 ;
				float64_t fixedtempvv_ = CMath::INFTY ;
				int32_t fixedtempii_ = 0 ;
				
			
				for (int32_t i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					
					const CPlifBase * penalty = PEN.element(j,ii) ;
					int32_t look_back = max_look_back ;
					{ // find lookback length
						CPlifBase *pen = (CPlifBase*) penalty ;
						if (pen!=NULL)
							look_back=(int32_t) (CMath::ceil(pen->get_max_value()));
						if (look_back>=1e6)
							SG_PRINT("%i,%i -> %d from %ld\n", j, ii, look_back, (long)pen) ;
						ASSERT(look_back<1e6);
					}
					//int32_t num_current_svms = trans_matrix_num_svms.element(j,ii);
					//int32_t* svm_ids = trans_matrix_svms.element(j,ii);
					//int32_t* svm_ids[num_current_svms];
					//for (int32_t id=0;id<num_current_svms;id++)
					//	svm_ids[id]=*(p_svm_ids+id);
				
					int32_t orf_from = orf_info.element(ii,0) ;
					int32_t orf_to   = orf_info.element(j,1) ;
					if((orf_from!=-1)!=(orf_to!=-1))
						SG_DEBUG("j=%i  ii=%i  orf_from=%i orf_to=%i p=%1.2f\n", j, ii, orf_from, orf_to, elem_val[i]) ;
					ASSERT((orf_from!=-1)==(orf_to!=-1)) ;
					
					int32_t orf_target = -1 ;
					if (orf_from!=-1)
					{
						orf_target=orf_to-orf_from ;
						if (orf_target<0) 
							orf_target+=3 ;
						ASSERT(orf_target>=0 && orf_target<3) ;
					}
					
					int32_t orf_last_pos = pos[t] ;
					int32_t loss_last_pos = t ;
					float64_t last_loss = 0.0 ;

					for (int32_t ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						bool ok ;
						//int32_t plen=t-ts;

						/*for (int32_t s=0; s<num_svms; s++)
							if ((fabs(svs.svm_values[s*svs.seqlen+plen]-svs2.svm_values[s*svs.seqlen+plen])>1e-6) ||
								(fabs(svs.svm_values[s*svs.seqlen+plen]-svs3.svm_values[s*svs.seqlen+plen])>1e-6))
							{
								SG_DEBUG( "s=%i, t=%i, ts=%i, %1.5e, %1.5e, %1.5e\n", s, t, ts, svs.svm_values[s*svs.seqlen+plen], svs2.svm_values[s*svs.seqlen+plen], svs3.svm_values[s*svs.seqlen+plen]);
								}*/
						
						if (orf_target==-1)
							ok=true ;
						else if (pos[ts]!=-1 && (pos[t]-pos[ts])%3==orf_target)
							ok=(!use_orf) || extend_orf(orf_from, orf_to, pos[ts], orf_last_pos, pos[t]) ;
						else
							ok=false ;
						
						if (ok)
						{
							
							float64_t segment_loss = 0.0 ;
							if (with_loss)
								segment_loss = extend_segment_loss(loss, pos, elem_id[i], ts, loss_last_pos, last_loss) ;

							////////////////////////////////////////////////////////
							// BEST_PATH_TRANS
							////////////////////////////////////////////////////////
							int32_t frame = orf_info.element(ii,0);
							lookup_content_svm_values(ts, t, pos[ts], pos[t], svm_value, frame);

							//int32_t offset = plen*num_svms ;
							//for (int32_t ss=0; ss<num_svms; ss++)
							//{
							//	//svm_value[ss]=svs.svm_values[offset+ss];
							//	svm_value[ss]=new_svm_value[ss];
							//	//if (CMath::abs(new_svm_value[ss]-svm_value[ss])>1e-5)
							//	//	SG_PRINT("ts: %i t: %i  precomp: %f old: %f diff: %f \n",ts, t,new_svm_value[ss],svm_value[ss], CMath::abs(new_svm_value[ss]-svm_value[ss]));
							//}

							float64_t pen_val = 0.0 ;
							if (penalty)
								pen_val = penalty->lookup_penalty(pos[t]-pos[ts], svm_value) ;
							if (nbest==1)
							{
								float64_t  val        = elem_val[i] + pen_val ;
								if (with_loss)
									val              += segment_loss ;
								
								float64_t mval = -(val + delta.element(delta_array, ts, ii, 0, seq_len, N)) ;
								if (mval<fixedtempvv_)
								{
									fixedtempvv_ = mval ;
									fixedtempii_ = ii + ts*N;
									fixed_list_len = 1 ;
								}
							}
							else
							{
								for (int16_t diff=0; diff<nbest; diff++)
								{
									float64_t  val        = elem_val[i]  ;
									val              += pen_val ;
									if (with_loss)
										val              += segment_loss ;
									
									float64_t mval = -(val + delta.element(delta_array, ts, ii, diff, seq_len, N)) ;
									
									/* only place -val in fixedtempvv if it is one of the nbest lowest values in there */
									/* fixedtempvv[i], i=0:nbest-1, is sorted so that fixedtempvv[0] <= fixedtempvv[1] <= ...*/
									/* fixed_list_len has the number of elements in fixedtempvv */
									
									if ((fixed_list_len < nbest) || ((0==fixed_list_len) || (mval < fixedtempvv[fixed_list_len-1])))
									{
										if ( (fixed_list_len<nbest) && ((0==fixed_list_len) || (mval>fixedtempvv[fixed_list_len-1])) )
										{
											fixedtempvv[fixed_list_len] = mval ;
											fixedtempii[fixed_list_len] = ii + diff*N + ts*N*nbest;
											fixed_list_len++ ;
										}
										else  // must have mval < fixedtempvv[fixed_list_len-1]
										{
											int32_t addhere = fixed_list_len;
											while ((addhere > 0) && (mval < fixedtempvv[addhere-1]))
												addhere--;
											
											// move everything from addhere+1 one forward 
											for (int32_t jj=fixed_list_len-1; jj>addhere; jj--)
											{
												fixedtempvv[jj] = fixedtempvv[jj-1];
												fixedtempii[jj] = fixedtempii[jj-1];
											}
											
											fixedtempvv[addhere] = mval;
											fixedtempii[addhere] = ii + diff*N + ts*N*nbest;
											
											if (fixed_list_len < nbest)
												fixed_list_len++;
										}
									}
								}
							}
						}
					}
				}
				
				
				int32_t numEnt = fixed_list_len;
				
				float64_t minusscore;
				int64_t fromtjk;
				
				for (int16_t k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
						if (nbest==1)
						{
							minusscore = fixedtempvv_ ;
							fromtjk = fixedtempii_ ;
						}
						else
						{
							minusscore = fixedtempvv[k];
							fromtjk = fixedtempii[k];
						}
						
					    delta.element(delta_array, t, j, k, seq_len, N)    = -minusscore + seq.element(j,t);
					    psi.element(t,j,k)      = (fromtjk%N) ;
						if (nbest>1)
							ktable.element(t,j,k)   = (fromtjk%(N*nbest)-psi.element(t,j,k))/N ;
					    ptable.element(t,j,k)   = (fromtjk-(fromtjk%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(delta_array, t, j, k, seq_len, N)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						if (nbest>1)
							ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
			}
		}
	}

	//clear_svm_values(svs);
	if (with_loss)
		clear_segment_loss(loss);

	{ //termination
		int32_t list_len = 0 ;
		for (int16_t diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				oldtempvv[list_len] = -(delta.element(delta_array, (seq_len-1), i, diff, seq_len, N)+get_q(i)) ;
				oldtempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(oldtempvv.get_array(), oldtempii.get_array(), list_len, nbest) ;
		
		for (int16_t k=0; k<nbest; k++)
		{
			delta_end.element(k) = -oldtempvv[k] ;
			path_ends.element(k) = (oldtempii[k]%N) ;
			if (nbest>1)
				ktable_end.element(k) = (oldtempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (int16_t k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			int32_t i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			int16_t q   = 0 ;
			if (nbest>1)
				q=ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				if (nbest>1)
					q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			int32_t num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*seq_len]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*seq_len]=-1 ;
			my_pos_seq[num_states+k*seq_len]=-1 ;
		}
	}
	
	if (is_big)
		SG_PRINT( "DONE.     \n") ;


#ifdef DYNPROG_TIMING
	MyTime2.stop() ;
	
	if (is_big)
		SG_PRINT("Timing:  orf=%1.2f s \n Segment_init=%1.2f s Segment_pos=%1.2f s  Segment_extend=%1.2f s Segment_clean=%1.2f s\nsvm_init=%1.2f s  svm_pos=%1.2f  svm_clean=%1.2f\n  total=%1.2f\n", orf_time, segment_init_time, segment_pos_time, segment_extend_time, segment_clean_time, svm_init_time, svm_pos_time, svm_clean_time, MyTime2.time_diff_sec()) ;
#endif

	delete[] fixedtempvv ;
	delete[] fixedtempii ;
}

void CDynProg::best_path_trans_deriv(
	int32_t *my_state_seq, int32_t *my_pos_seq, float64_t *my_scores,
	float64_t* my_losses,int32_t my_seq_len, const float64_t *seq_array,
	int32_t seq_len, const int32_t *pos, CPlifBase **Plif_matrix,
	CPlifBase **Plif_state_signals, int32_t max_num_signals,int32_t genestr_num)
{	
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	} ;
	//SG_PRINT( "genestr_len=%i, genestr_num=%i\n", genestr_len, genestr_num) ;
	//mod_words.display() ;
	//sign_words.display() ;
	//string_words.display() ;

	bool use_svm = false ;
	//ASSERT(dict_len==num_svms*cum_num_words_array[num_degrees]) ;
	//dict_weights.set_array(dictionary_weights, cum_num_words_array[num_degrees], num_svms, false, false) ;
	//dict_weights_array=dict_weights.get_array() ;
	
	CArray2<CPlifBase*> PEN(Plif_matrix, N, N, false, false) ;
	CArray2<CPlifBase*> PEN_state_signals(Plif_state_signals, N, max_num_signals, false, false) ;
	CArray3<float64_t> seq_input(seq_array, N, seq_len, max_num_signals) ;
	
	{ // determine whether to use svm outputs and clear derivatives
		for (int32_t i=0; i<N; i++)
			for (int32_t j=0; j<N; j++)
			{
				CPlifBase *penij=PEN.element(i,j) ;
				if (penij==NULL)
					continue ;
				
				if (penij->uses_svm_values())
					use_svm=true ;
				penij->penalty_clear_derivative() ;
			}
		for (int32_t i=0; i<N; i++)
			for (int32_t j=0; j<max_num_signals; j++)
			{
				CPlifBase *penij=PEN_state_signals.element(i,j) ;
				if (penij==NULL)
					continue ;
				if (penij->uses_svm_values())
					use_svm=true ;
				penij->penalty_clear_derivative() ;
			}
	}

	{ // set derivatives of p, q and a to zero
		for (int32_t i=0; i<N; i++)
		{
			initial_state_distribution_p_deriv.element(i)=0 ;
			end_state_distribution_q_deriv.element(i)=0 ;
			for (int32_t j=0; j<N; j++)
				transition_matrix_a_deriv.element(i,j)=0 ;
		}
	}
	
	{ // clear score vector
		for (int32_t i=0; i<my_seq_len; i++)
		{
			my_scores[i]=0.0 ;
			my_losses[i]=0.0 ;
		}
	}

	//int32_t total_len = 0 ;
	
	//transition_matrix_a.display_array() ;
	//transition_matrix_a_id.display_array() ;
	
	{ // compute derivatives for given path
		float64_t* svm_value = new float64_t[m_num_lin_feat_plifs_cum[m_num_raw_data]];
		for (int32_t s=0; s<m_num_lin_feat_plifs_cum[m_num_raw_data]; s++)
			svm_value[s]=0 ;
		
		for (int32_t i=0; i<my_seq_len; i++)
		{
			my_scores[i]=0.0 ;
			my_losses[i]=0.0 ;
		}
		
//#ifdef DYNPROG_DEBUG
		float64_t total_score = 0.0 ;
		float64_t total_loss = 0.0 ;
//#endif		

		ASSERT(my_state_seq[0]>=0) ;
		initial_state_distribution_p_deriv.element(my_state_seq[0])++ ;
		my_scores[0] += initial_state_distribution_p.element(my_state_seq[0]) ;

		ASSERT(my_state_seq[my_seq_len-1]>=0) ;
		end_state_distribution_q_deriv.element(my_state_seq[my_seq_len-1])++ ;
		my_scores[my_seq_len-1] += end_state_distribution_q.element(my_state_seq[my_seq_len-1]);

//#ifdef DYNPROG_DEBUG
		total_score += my_scores[0] + my_scores[my_seq_len-1] ;
//#endif 		
		struct svm_values_struct svs;
		svs.num_unique_words = NULL;
		svs.svm_values = NULL;
		svs.svm_values_unnormalized = NULL;
		svs.word_used = NULL;
		//init_svm_values(svs, my_state_seq[0], seq_len, 100);

		struct segment_loss_struct loss;
		loss.segments_changed = NULL;
		loss.num_segment_id = NULL;
		//SG_DEBUG( "seq_len=%i\n", my_seq_len) ;
		for (int32_t i=0; i<my_seq_len-1; i++)
		{
			if (my_state_seq[i+1]==-1)
				break ;
			int32_t from_state = my_state_seq[i] ;
			int32_t to_state   = my_state_seq[i+1] ;
			int32_t from_pos   = my_pos_seq[i] ;
			int32_t to_pos     = my_pos_seq[i+1] ;

			// compute loss relative to another segmentation using the segment_loss function
			init_segment_loss(loss, seq_len, pos[to_pos]-pos[from_pos]+10);
			find_segment_loss_till_pos(pos, to_pos,m_segment_ids, m_segment_mask, loss);  

			int32_t loss_last_pos = to_pos ;
			float64_t last_loss = 0.0 ;
			int32_t elem_id = transition_matrix_a_id.element(from_state, to_state) ;
			my_losses[i] = extend_segment_loss(loss, pos, elem_id, from_pos, loss_last_pos, last_loss) ;
#ifdef DYNPROG_DEBUG
			io->set_loglevel(M_DEBUG) ;
			SG_DEBUG( "%i. segment loss %f (id=%i): from=%i(%i), to=%i(%i)\n", i, my_losses[i], elem_id, from_pos, from_state, to_pos, to_state) ;
#endif
			// increase usage of this transition
			transition_matrix_a_deriv.element(from_state, to_state)++ ;
			my_scores[i] += transition_matrix_a.element(from_state, to_state) ;
			//SG_PRINT("transition_matrix_a.element(%i, %i),%f \n",from_state, to_state, transition_matrix_a.element(from_state, to_state));
#ifdef DYNPROG_DEBUG
			SG_DEBUG( "%i. scores[i]=%f\n", i, my_scores[i]) ;
#endif
			
			/*int32_t last_svm_pos[num_degrees] ;
			for (int32_t qq=0; qq<num_degrees; qq++)
			last_svm_pos[qq]=-1 ;*/
			
			if (use_svm)
			{
				//SG_PRINT("from_pos: %i; to_pos: %i; pos[to_pos]-pos[from_pos]: %i \n",from_pos, to_pos, pos[to_pos]-pos[from_pos]); 
				int32_t frame = m_orf_info.element(from_state,0);
				if (false)//(frame>=0)
				{
					int32_t num_current_svms=0;
					int32_t svm_ids[] = {-8, -7, -6, -5, -4, -3, -2, -1};
					SG_PRINT("penalties(%i, %i), frame:%i  ", from_state, to_state, frame);
					PEN.element(to_state, from_state)->get_used_svms(&num_current_svms, svm_ids);
					SG_PRINT("\n");
				}


				lookup_content_svm_values(from_pos, to_pos, pos[from_pos],pos[to_pos], svm_value, frame);
				if (false)//(frame>=0)
					SG_PRINT("svm_values: %f, %f, %f \n", svm_value[4], svm_value[5], svm_value[6]);
				//SG_PRINT("svm_values: %f, %f, %f, %f \n", svm_value[8], svm_value[9], svm_value[10], svm_value[11]);
	
				//#ifdef DYNPROG_DEBUG
				//					SG_DEBUG( "svm[%i]: %f\n", ss, svm_value[ss]) ;
				//#endif
			}
			
			if (PEN.element(to_state, from_state)!=NULL)
			{
				float64_t nscore = PEN.element(to_state, from_state)->lookup_penalty(pos[to_pos]-pos[from_pos], svm_value) ;
				my_scores[i] += nscore ;

				for (int32_t s=num_svms;s<m_num_lin_feat_plifs_cum[m_num_raw_data];s++)/*set tiling plif values to neutral values (that do not influence derivative calculation)*/
					svm_value[s]=-CMath::INFTY;
			
#ifdef DYNPROG_DEBUG
				//SG_DEBUG( "%i. transition penalty: from_state=%i to_state=%i from_pos=%i [%i] to_pos=%i [%i] value=%i\n", i, from_state, to_state, from_pos, pos[from_pos], to_pos, pos[to_pos], pos[to_pos]-pos[from_pos]) ;

				/*
				  int32_t orf_from = g_orf_info.element(from_state,0) ;
				  int32_t orf_to   = g_orf_info.element(to_state,1) ;
				  ASSERT((orf_from!=-1)==(orf_to!=-1)) ;
				  if (orf_from != -1)
				  total_len = total_len + pos[to_pos]-pos[from_pos] ;
				  
				  SG_DEBUG( "%i. orf_info: from_orf=%i to_orf=%i orf_diff=%i, len=%i, lenmod3=%i, total_len=%i, total_lenmod3=%i\n", i, orf_from, orf_to, (orf_to-orf_from)%3, pos[to_pos]-pos[from_pos], (pos[to_pos]-pos[from_pos])%3, total_len, total_len%3) ;
				*/
#endif
				PEN.element(to_state, from_state)->penalty_add_derivative(pos[to_pos]-pos[from_pos], svm_value) ;
				for (int32_t d=1;d<=m_num_raw_data;d++) 
				{
					for (int32_t s=0;s<m_num_lin_feat_plifs_cum[m_num_raw_data];s++)
						svm_value[s]=-CMath::INFTY;
					float64_t* intensities = new float64_t[m_num_probes_cum[d]];
					int32_t num_intensities = raw_intensities_interval_query(pos[from_pos], pos[to_pos],intensities, d);
					//SG_PRINT("pos[from_pos]:%i, pos[to_pos]:%i, num_intensities:%i\n",pos[from_pos],pos[to_pos], num_intensities);
					for (int32_t k=0;k<num_intensities;k++)
					{
					 	for (int32_t j=m_num_lin_feat_plifs_cum[d-1];j<m_num_lin_feat_plifs_cum[d];j++)
							svm_value[j]=intensities[k];
						
						PEN.element(to_state, from_state)->penalty_add_derivative(-CMath::INFTY, svm_value) ;	
						
					}
					delete[] intensities;
				}
			}
#ifdef DYNPROG_DEBUG
			SG_DEBUG( "%i. scores[i]=%f\n", i, my_scores[i]) ;
#endif

			//SG_DEBUG( "emmission penalty skipped: to_state=%i to_pos=%i value=%1.2f score=%1.2f\n", to_state, to_pos, seq_input.element(to_state, to_pos), 0.0) ;
			for (int32_t k=0; k<max_num_signals; k++)
			{
				if ((PEN_state_signals.element(to_state,k)==NULL)&&(k==0))
				{
#ifdef DYNPROG_DEBUG
					SG_DEBUG( "%i. emmission penalty: to_state=%i to_pos=%i score=%1.2f (no signal plif)\n", i, to_state, to_pos, seq_input.element(to_state, to_pos, k)) ;
#endif
					my_scores[i] += seq_input.element(to_state, to_pos, k) ;
					//if (seq_input.element(to_state, to_pos, k) !=0)
					//	SG_PRINT("features(%i,%i): %f\n",to_state,to_pos,seq_input.element(to_state, to_pos, k));
					break ;
				}
				if (PEN_state_signals.element(to_state, k)!=NULL)
				{
					float64_t nscore = PEN_state_signals.element(to_state,k)->lookup_penalty(seq_input.element(to_state, to_pos, k), svm_value) ;
					my_scores[i] += nscore ;
					//break ;
					//int32_t num_current_svms=0;
					//int32_t svm_ids[] = {-8, -7, -6, -5, -4, -3, -2, -1};
					//SG_PRINT("PEN_state_signals->id: ");
					//PEN_state_signals.element(to_state, k)->get_used_svms(&num_current_svms, svm_ids);
					//SG_PRINT("\n");
					//if (nscore != 0)
						//SG_PRINT( "%i. emmission penalty: to_state=%i to_pos=%i value=%1.2f score=%1.2f k=%i\n", i, to_state, to_pos, seq_input.element(to_state, to_pos, k), nscore, k) ;
#ifdef DYNPROG_DEBUG
					SG_DEBUG( "%i. emmission penalty: to_state=%i to_pos=%i value=%1.2f score=%1.2f k=%i\n", i, to_state, to_pos, seq_input.element(to_state, to_pos, k), nscore, k) ;
#endif
					PEN_state_signals.element(to_state,k)->penalty_add_derivative(seq_input.element(to_state, to_pos, k), svm_value) ;
				} else
					break ;
			}
			
//#ifdef DYNPROG_DEBUG
			//SG_PRINT( "scores[%i]=%f (final) \n", i, my_scores[i]) ;
			//SG_PRINT( "losses[%i]=%f (final) , total_loss: %f \n", i, my_losses[i], total_loss) ;
			total_score += my_scores[i] ;
			total_loss += my_losses[i] ;
//#endif
		}
//#ifdef DYNPROG_DEBUG
		//SG_PRINT( "total score = %f \n", total_score) ;
		SG_PRINT( "total loss = %f \n", total_loss) ;
//#endif
		clear_svm_values(svs);
		clear_segment_loss(loss);
		delete[] svm_value;
	}

	
}


void CDynProg::best_path_trans_simple(
	const float64_t *seq_array, int32_t seq_len, int16_t nbest,
	float64_t *prob_nbest, int32_t *my_state_seq)
{
	if (!svm_arrays_clean)
	{
		SG_ERROR( "SVM arrays not clean") ;
		return ;
	} ;

	int32_t max_look_back = 2 ;
	const int32_t look_back_buflen = max_look_back*nbest*N ;
	ASSERT(nbest<32000) ;
		
	CArray2<float64_t> seq((float64_t *)seq_array, N, seq_len, false) ;

	CArray3<float64_t> delta(max_look_back, N, nbest) ;
	CArray3<T_STATES> psi(seq_len, N, nbest) ;
	CArray3<int16_t> ktable(seq_len,N,nbest) ;
	CArray3<int32_t> ptable(seq_len,N,nbest) ;

	CArray<float64_t> delta_end(nbest) ;
	CArray<T_STATES> path_ends(nbest) ;
	CArray<int16_t> ktable_end(nbest) ;

	CArray<float64_t> oldtempvv(look_back_buflen) ;
	CArray<int32_t> oldtempii(look_back_buflen) ;

	CArray<T_STATES> state_seq(seq_len) ;
	CArray<int32_t> pos_seq(seq_len) ;

	{ // initialization

		for (T_STATES i=0; i<N; i++)
		{
			delta.element(0,i,0) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			psi.element(0,i,0)   = 0 ;
			ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (int16_t k=1; k<nbest; k++)
			{
				delta.element(0,i,k)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;                  // <--- what's this for?
				ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (int32_t t=1; t<seq_len; t++)
	{
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (int16_t k=0; k<nbest; k++)
				{
					delta.element(t%max_look_back,j,k)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const float64_t *elem_val      = trans_list_forward_val[j] ;
				
				int32_t old_list_len = 0 ;
				
				for (int32_t i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;

					int32_t ts=t-1; 
					if (ts>=0)
					{
						bool ok=true ;
						
						if (ok)
						{

						  
						  for (int16_t diff=0; diff<nbest; diff++)
						    {
						      float64_t  val        = delta.element(ts%max_look_back,ii,diff) + elem_val[i] ;
						      float64_t mval = -val;

						      oldtempvv[old_list_len] = mval ;
						      oldtempii[old_list_len] = ii + diff*N + ts*N*nbest;
						      old_list_len++ ;
						    }
						}
					}
				}
				
				CMath::nmin<int32_t>(oldtempvv.get_array(), oldtempii.get_array(), old_list_len, nbest) ;

				int32_t numEnt = 0;
				numEnt = old_list_len;

				float64_t minusscore;
				int64_t fromtjk;

				for (int16_t k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
					    minusscore = oldtempvv[k];
					    fromtjk = oldtempii[k];
					    
					    delta.element(t%max_look_back,j,k)    = -minusscore + seq.element(j,t);
					    psi.element(t,j,k)      = (fromtjk%N) ;
					    ktable.element(t,j,k)     = (fromtjk%(N*nbest)-psi.element(t,j,k))/N ;
					    ptable.element(t,j,k)     = (fromtjk-(fromtjk%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(t%max_look_back,j,k)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
				
			}
		}
	}

	
	{ //termination
		int32_t list_len = 0 ;
		for (int16_t diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				oldtempvv[list_len] = -(delta.element((seq_len-1)%max_look_back,i,diff)+get_q(i)) ;
				oldtempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(oldtempvv.get_array(), oldtempii.get_array(), list_len, nbest) ;
		
		for (int16_t k=0; k<nbest; k++)
		{
			delta_end.element(k) = -oldtempvv[k] ;
			path_ends.element(k) = (oldtempii[k]%N) ;
			ktable_end.element(k) = (oldtempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (int16_t k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			int32_t i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			int16_t q   = ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			int32_t num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
			}
			//my_state_seq[num_states+k*seq_len]=-1 ;
		}

	}
}

