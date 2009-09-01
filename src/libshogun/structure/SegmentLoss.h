#ifndef __SEGMENT_LOSS__
#define __SEGMENT_LOSS__

#include "lib/common.h"     
#include "base/SGObject.h"
                   
/** @brief class IntronList */   
class CSegmentLoss : public CSGObject
{                         
        public:
                CSegmentLoss();
                virtual ~CSegmentLoss();

		float64_t get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id);

		/** set best path segment loss
		 *
		 * @param segment_loss segment loss
		 * @param num_segment_id1 number of segment id1
		 * @param num_segment_id2 number of segment id2
		 */
		void best_path_set_segment_loss(float64_t* segment_loss, int32_t m, int32_t n);

		/** set best path segmend ids mask
		 *
		 * @param segment_ids segment ids
		 * @param segment_mask segment mask
		 * @param m dimension m
		 */
		void best_path_set_segment_ids_mask(int32_t* segment_ids, float64_t* segment_mask, int32_t m);

		
		inline virtual const char* get_name() const { return "SegmentLoss"; }
	protected:             
};
#endif                 

