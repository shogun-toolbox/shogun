#ifndef __OBSERVATION_H_
#define __OBSERVATION_H_

#include <stdio.h>
#include "lib/io.h"
#include "lib/common.h"
#include "lib/Mathmatics.h"

/*
typedef struct word {
  FNUM    wnum;	
  FVAL    weight;
} WORD;*/


typedef struct doc {
  long    docnum;
  double  twonorm_sq;
//  WORD    *words;
} DOC;

//define numbers for the bases 
const unsigned char B_A=0;
const unsigned char B_C=1;
const unsigned char B_G=2;
const unsigned char B_T=3;
const unsigned char B_N=4;
const unsigned char B_n=5;

/** type that is used for observations - can be BYTE or WORD.
 * Probably BYTE is enough if you have at most 256 observations
 * however WORD is also possible although you might quickly run into memory problems
 * NOTE: If you want to use higher order HMMs this type must hold values up to \#observations**ORDER
 */
#ifdef BIGOBS
typedef WORD* P_OBSERVATIONS ;
typedef WORD T_MAPTABLE;
typedef WORD T_OBSERVATIONS ;
#else
typedef BYTE* P_OBSERVATIONS ;
typedef BYTE T_MAPTABLE;
typedef BYTE T_OBSERVATIONS ;
#endif 

/// Type of observations
enum E_OBS_TYPE
{
	/// Train samples of classes +1/-1
	TRAIN,

	/// Samples of class +1
	POSTRAIN,

	/// Samples of class -1
	NEGTRAIN,

	/// Test samples of class +/-1
	TEST,

	/// Test Samples for class +1
	POSTEST,

	/// Test Samples for class -1
	NEGTEST,

	/// Unlabeled Samples
	UNLABELED,

	/// Default
	NONE
};

/// Alphabet of observations
enum E_OBS_ALPHABET
{
	/// DNA - letters A,C,G,T
	DNA=0,

	/// PROTEIN - letters a-z
	PROTEIN=1,

	/// ALPHANUM - [0-9a-z]
	ALPHANUM=2,

	/// CUBE - [1-6]
	CUBE=3
};


/** Observation related Functions.
 * Class which collects generic observation functions
 */
class CObservation  
{

public:
	
	/**@name Constructor/Destructor.
	 */
	//@{
	/** create an empty observation matrix 
	 * and initialize arrays
	 */
	CObservation(E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER=1);

	/// combine positive and negative observations
	CObservation(CObservation* pos, CObservation* neg);


	/** read an observation matrix into memory and translates the read characters.
	 * \begin{verbatim}
	  -format specs: in_file (gene.lin)
	  ([AGCT]+<<EOL|EOF>>)+
	  \end{verbatim}
	 * a row of the observation matrix corresponds to one dimension of input.
	 * @param file filehandle to observations
	 */
	CObservation(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER=1);

	///Destructor - frees logtable
	virtual ~CObservation();
	//@}

	/** read an observation matrix into memory and translates the read characters.
	 * \begin{verbatim}
	  -format specs: in_file (gene.lin)
	  ([AGCT]+<<EOL|EOF>>)+
	  \end{verbatim}
	 * a row of the observation matrix corresponds to one dimension of input.
	 * @param file filehandle to observations
	 */
	bool load_observations(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER=1);

	/** add num_sv support vectors to observations
	 * \begin{verbatim}
	  -format specs: in_file (machine.svm)
	  (<NUMBER>:[AGCT]+<<EOL|EOF>>)+
	  \end{verbatim}
	 * @param file filehandle to supportvector machine
	 */
	bool add_support_vectors(FILE* file, int num_sv);

	/// get index of supportvector in observations
	inline int get_support_vector_idx(int sv) { return sv_idx+sv; }
	
	/// get number of supportvectors in observations
	inline int get_support_vector_num() { return sv_num; }

	///free observations
	void cleanup();

	/** access function for observations
	 * @param dimension dimension of observations 0...DIMENSION-1
	 * @param time 0...T-1
	 * @return observation 
	 */
	inline T_OBSERVATIONS get_obs(int dimension, int time) const
	{
	    return observations[dimension][time];
	}

	/** access function for observations
	 * @param dimension dimension of observations 0...DIMENSION-1
	 * @return specified dimension of observation matrix (a vector: o_o,...,o_T-1)
	 */
	inline T_OBSERVATIONS* get_obs_vector(int dimension)
	{
	    return observations[dimension];
	}

	/** get maximum number of columns of observation matrix
	 * @return maximum length of observation vectors
	 */
	inline int get_obs_max_T() const 
	{
		return max_T;
	}

	/** get inherited size of emissions type of HMM
	 * @return MAX_M
	 */
	inline int get_max_M() const
	{
	    return MAX_M;
	}
	
	/** get inherited number of emissions of HMM
	 * @return M
	 */
	inline int get_M() const
	{
	    return M;
	}

	/** get number of columns of vector at dimension in observation matrix
	 * @return length of observation vector
	 */
	inline int get_obs_T(int dimension) const
	{
	    return observation_len[dimension];
	}

	/** get label of observation row vector
	 * @return label of observation
	 */
	inline int get_label(int dimension) const
	{
		if (label!=NULL) //if (observation_type!=UNLABELED)
			return label[dimension];
		if (observation_type==POSTRAIN || observation_type==POSTEST)
			return +1;
		else if (observation_type==NEGTRAIN || observation_type==NEGTEST)
			return -1;
		else 
			return 0;
	}

	/** access function for observations
	 * @param dimension dimension of observations 0...DIMENSION-1
	 * @param time 0...T-1
	 * @parma value value to be set
	 */
	inline void set_obs(int dimension, int time, T_OBSERVATIONS value) 
	{
	    observations[dimension][time]=value ;
	}
	
	/** set length of the observations vector at dimension
	 * @param dim dimension of observations 0...DIMENSION-1
	 * @param length length of vector
	 */
	inline void set_obs_len(int dim, int length) 
	{
	    observation_len[dim]=length ;
	}

	/// return observation type
	inline E_OBS_TYPE get_type()
	{
	    return observation_type;
	}

	/// return working alphabet
	inline E_OBS_ALPHABET get_alphabet()
	{
	    return alphabet_type;
	}

	/// get the number of dimensions
	inline int get_DIMENSION() 
	{
	    return DIMENSION;
	}

	/// reduce the number of dimensions to dim
	inline void set_dimension(int dim) 
	{
	    DIMENSION=math.min(dim,REALDIMENSION) ;
	    CIO::message("setting number of patterns to %i/%i (virtually)\n", DIMENSION,REALDIMENSION) ;
	}

	/// remap observations e.g translate ACGT to 0123
	inline T_OBSERVATIONS remap(T_OBSERVATIONS obs)
	{
	    return maptable[obs];
	}

	/** translates e.g. ACGT to 0123 and maps higher order models to the alphabet
	 * @param observations observations
	 * @param sequence_length length of observation vector
	 */
	int translate_from_single_order(T_OBSERVATIONS* observations_, int sequence_length);

	/** translates e.g. 0123 to ACGT and drops higher dimensions of emissions
	 * @param observations observations
	 * @param sequence_length length of observation vector
	 */
	void translate_to_single_order(T_OBSERVATIONS* observations, int sequence_length);

protected:
	/// initialize maptable which is used for fast character translation
	void init_map_table();

protected:
	/// observation matrix
	T_OBSERVATIONS** observations;

	/// sizes of observations (== #columns of observation matrix)
	int* observation_len;

	/// class belongings
	int* label;

	/// chosen alphabet
	E_OBS_ALPHABET alphabet_type;

	/// type of observation
	E_OBS_TYPE observation_type;

	/// virtual DIMENSION of observation matrix (==number of observation vectors)
	int DIMENSION;

	/// real DIMENSION of observation matrix (==number of observation vectors)
	int REALDIMENSION;

	/**@name Observation translation
	 * maps ACGT -> 0123
	 * and variables necessary for higher order mapping to alphabet
	 */
	//@{
	/// Translation Table ACGT -> 0123 und 0123->ACGT
	static T_OBSERVATIONS maptable[1 << (sizeof(T_MAPTABLE)*8)];

	/// size of M
	int MAX_M;
	
	/// maximal # of observations
	int M;

	/// order
	int ORDER;

	/// maximum #columns of observation matrix
	int max_T;

	/// number of support vectors added to data
	int sv_num;
	
	/// index of first support vector
	int sv_idx;

	/// content of file
	char* full_content;
	//@}
};
#endif
