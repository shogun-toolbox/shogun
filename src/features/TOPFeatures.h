#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/Features.h"

double* CHMM::feature_cache_sv=NULL;	
double* CHMM::feature_cache_obs=NULL;	
int CHMM::num_features=0;	
bool CHMM::feature_cache_in_question=false;	
unsigned int CHMM::feature_cache_checksums[8];	
unsigned int CHMM::features_crc32[32];
class CTOPFeatures::CFeatures
{
	/// state of top-feature cache
	enum E_TOP_FEATURE_CACHE_VALIDITY
	{
	    VALID=0,
	    OBS_INVALID=1,
	    SV_INVALID=2,
	    QUESTIONABLE=3,
	    INVALID=4
	};
	~CFeatures();

	/** set feature matrix
		necessary to set feature_matrix, num_features, num_vectors, where
		num_features is the column offset, and columns are linear in memory
		see below for definition of feature_matrix
	*/
	void set_feature_matrix();
	/**@name arrays and caches.
	 * arrays for model parameters
	 * caches for forward/backward variables, feature cache etc
	 */
	//@{
	
	// feature cache for observations
	static double* feature_cache_obs;	
	// feature cache for support vectors
	static double* feature_cache_sv;

	// feature cache validity needs to be checked if set
	static bool feature_cache_in_question;	
	// feature cache checksums with wich validity is checked
	static unsigned int feature_cache_checksums[8];	
	// checksums on observations to see whether cache is still valid
	static unsigned int features_crc32[32];
	// number of features per cacheline
	static int num_features;	
	/**@name functions for TOP feature vector
	 * computes the TOP featurevector
	 * @param pos positive model
	 * @param neg negative model
	 */ 
	//@{
	   
	    /// compute featurevectors for all observations and return a cache of size num_features*num_observations
	    static bool compute_top_feature_cache(CHMM* pos, CHMM* neg);

		/// save all top featurevectors into dest
		static bool save_top_features(CHMM* pos, CHMM* neg, FILE* dest);
	    
	    /**@name compute featurevector for observation dim
	     * Computes the featurevector for one observation 
	     * @param dim specifies the observation for which the featurevector is calculated
	     * @param featurevector if not NULL the vector will be written to that address
	     * @return returns the featurevector or NULL if unsuccessfull
	     */ 
	    static double* compute_top_feature_vector(CHMM* pos, CHMM* neg, int dim, double* featurevector=NULL);
	    
	    /**@name get number of features
	     * @return returns the number of features in a feature cache line
	     */ 
	    static inline int get_top_num_features() { return CHMM::num_features; }

	    /**@name get feature cache line
	     * @return returns a pointer to a line in featurecache specified by row
	     */ 
	    static inline double* get_top_feature_cache_line(int row)
	    {
		//feature_cache_checksums[6] stands for DIMENSION of observations - sorry for that hack
		if (row<(int) feature_cache_checksums[6])
		    return &feature_cache_obs[num_features*row];
		else
		    return &feature_cache_sv[num_features*(row - (int)feature_cache_checksums[6])];
	    }
		void subtract_mean_from_top_feature_cache(int num_features, int totobs);

	    ///invalidate the top feature cache
	    static void invalidate_top_feature_cache(E_TOP_FEATURE_CACHE_VALIDITY v);

	    ///update checksums of top feature cache
	    static void update_checksums(CObservation* o);

	    /**@name check for different observations
	     * @return returns 0 if crc is still ok, bit 1 is set for observation-change bit 2 for sv-change
	     */ 
	    static void check_and_update_crc(CHMM* pos, CHMM* neg);
	//@}
	
};
#endif
