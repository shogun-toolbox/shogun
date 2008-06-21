#ifndef KERNEL_OLIGO_H
#define KERNEL_OLIGO_H

#include "kernel/Kernel.h"
#include <vector>
#include <string>
#include <utility>

/**
  @brief This class offers access to the Oligo Kernel introduced 
  by Meinicke et al. in 2004

  The class has functions to preprocess the data such that the kernel
  computation can be pursued faster. The kernel function is then
  kernelOligoFast or kernelOligo.

*/
class COligoKernel : public CKernel
{
	public:
		/// Constructor
		COligoKernel(INT cache_size);
		/// Destructor
		~COligoKernel();

		/**
		  @brief encodes the signals of the sequence

		  This function stores the oligo function signals in 'values'. 
		  The 'k_mer_length' and the 'allowed_characters' determine, 
		  which signals are used. Every pair contains the position of the
		  signal and a numerical value reflecting the signal. The
		  numerical value represents the k_mer to a base 
		  n = |allowed_characters|. 

Example: The value of k_mer CG for the allowed characters ACGT
would be 1 * n^1 + 2 * n^0 = 6.
*/ 				
		static void encodeOligo(const std::string&                       sequence,
				unsigned int                             k_mer_length,
				const std::string&                       allowed_characters,
				std::vector< std::pair<int, double> >&   values);

		/**
		  @brief encodes all sequences with the encodeOligo function and stores
		  them in 'encoded_sequences'

		  This function encodes the sequences of 'sequences' via the
		  function encodeOligo.	        
		  */
		static void getSequences(const std::vector<std::string>& sequences, 
				unsigned int k_mer_length, 
				const std::string& allowed_characters, 
				std::vector< std::vector< std::pair<int, double> > >& encoded_sequences);

		/**
		  @brief prepares the exp function cache of the oligo kernel

		  The oligo kernel was introduced for sequences of fixed length.
		  Let n be the sequence length of sequences x and y. There can 
		  only be n different distances between signals in sequence x 
		  and sequence y (0, 1, ..., n-1). Therefore, we precompute 
		  the corresponding values of the e-function. These values 
		  can then be used in kernelOligoFast.
		  */
		static void getExpFunctionCache(double                sigma, 
				unsigned int          sequence_length, 
				std::vector<double>&  cache);

		/**
		  @brief returns the value of the oligo kernel for sequences 'x' and 'y'

		  This function computes the kernel value of the oligo kernel,
		  which was introduced by Meinicke et al. in 2004. 'x' and
		  'y' are encoded by encodeOligo and 'exp_cache' has to be 
		  constructed by getExpFunctionCache. 

		  'max_distance' can be used to speed up the computation 
		  even further by restricting the maximum distance between a k_mer at
		  position i in sequence 'x' and a k_mer at position j 
		  in sequence 'y'. If i - j > 'max_distance' the value is not
		  added to the kernel value. This approximation is switched
		  off by default (max_distance < 0).
		  */
		static double kernelOligoFast(const std::vector< std::pair<int, double> >&    x, 
				const std::vector< std::pair<int, double> >&    y,
				const std::vector<double>& 	                    exp_cache,
				int 			                    max_distance = -1);

		/**
		  @brief returns the value of the oligo kernel for sequences 'x' and 'y'

		  This function computes the kernel value of the oligo kernel,
		  which was introduced by Meinicke et al. in 2004. 'x' and
		  'y' have to be encoded by encodeOligo. 
		  */
		static double kernelOligo(const std::vector< std::pair<int, double> >&    x, 
				const std::vector< std::pair<int, double> >&    y,
				double 			                        sigma_square);

	private: 
		static bool cmpOligos_( std::pair<int, double> a, std::pair<int, double> b ); 

};
#endif // KERNEL_OLIGO_H
