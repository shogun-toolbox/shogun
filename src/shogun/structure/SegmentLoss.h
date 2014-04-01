/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Jonas Behr
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __SEGMENT_LOSS__
#define __SEGMENT_LOSS__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicArray.h>


namespace shogun
{
	template <class T> class CDynamicArray;
/** @brief class IntronList */
class CSegmentLoss : public CSGObject
{
        public:

		/** constructor
		 */
		CSegmentLoss();

		virtual ~CSegmentLoss();

		/** get segment loss for a given range
		 *
		 * @param from_pos start position
		 * @param to_pos end position
		 * @param segment_id type of the segment
		 */
		float32_t get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id);

		/** get segment loss for a given range
		 *
		 * @param from_pos start position
		 * @param to_pos end position
		 * @param segment_id type of the segment
		 */
		float32_t get_segment_loss_extend(int32_t from_pos, int32_t to_pos, int32_t segment_id);

		/** set best path segment loss
		 *
		 * @param segment_loss segment loss
		 * @param m number of segment id1
		 * @param n  number of segment id2
		 */
		void set_segment_loss(float64_t* segment_loss, int32_t m, int32_t n);

		/** set best path segmend ids
		 *
		 * @param segment_ids segment ids
		 */
		void set_segment_ids(CDynamicArray<int32_t>* segment_ids);

		/** mask parts of the sequence such that there is no
		 *  loss incured there; this is used if there is uncertainty
		 *  in the label
		 *
		 * @param segment_mask mask
		 */
		void set_segment_mask(CDynamicArray<float64_t>* segment_mask);

		/** set num segment types
		 *
		 * @param num_segment_types num segment types
		 */
		void set_num_segment_types(int32_t num_segment_types)
		{
			m_num_segment_types = num_segment_types;
		}

		/** compute loss
		 *
		 * @param all_pos all candidate positions
		 * @param len number of positions
		 */
		void compute_loss(int32_t* all_pos, int32_t len);

		/**
		 * @return object name
		 */
		virtual const char* get_name() const { return "SegmentLoss"; }
	protected:

		/** segment loss matrix*/
		CDynamicArray<float32_t> m_segment_loss_matrix; // 2d

		/** segment loss
		 *  two square matrices:
		 *  one for segment based loss and
		 *  one for length contribution*/
		CDynamicArray<float64_t> m_segment_loss; // 3d

		/** segment IDs */
		CDynamicArray<int32_t>* m_segment_ids;

		/** segment mask */
		CDynamicArray<float64_t>* m_segment_mask;

		/** number of different segment types (former: max_a_id)*/
		int32_t m_num_segment_types;
};

inline float32_t CSegmentLoss::get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{

	/*	int32_t from_pos_shift = from_pos ;
		if (print)
		SG_PRINT("# pos=%i,%i  segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n",
				 from_pos_shift, to_pos, segment_id,
				 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;
	while(1)
	{
		while (m_segment_ids->element(from_pos_shift)==m_segment_ids->element(from_pos_shift+1) && from_pos_shift<to_pos)
			from_pos_shift++ ;
		if (print)
			SG_PRINT("# pos=%i,%i  segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n",
					 from_pos_shift, to_pos, segment_id,
					 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;

		if (from_pos_shift>=to_pos)
		{
			//SG_PRINT("break")
			break ;
		}
		else from_pos_shift++ ;
		}
	if (print)
	SG_PRINT("break\n")  */

	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos)-m_segment_loss_matrix.element(segment_id, to_pos);
	diff_contrib += m_segment_mask->element(to_pos-1)*m_segment_loss.element(segment_id, m_segment_ids->element(to_pos-1), 0);
	return diff_contrib;
}

inline float32_t CSegmentLoss::get_segment_loss_extend(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{
	int32_t from_pos_shift = from_pos ;

	/*SG_PRINT("segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n",
			 segment_id,
			 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;*/

	while (from_pos_shift<to_pos && m_segment_ids->element(from_pos_shift)==m_segment_ids->element(from_pos_shift+1))
		from_pos_shift++ ;

	/*SG_PRINT("segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n",
			 segment_id,
			 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;*/

	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos);
	//diff_contrib += m_segment_mask->element(to_pos)*m_segment_loss.element(segment_id, m_segment_ids->element(to_pos), 0);

	//if (from_pos_shift!=from_pos)
	//	SG_PRINT("shifting from %i to %i, to_pos=%i, loss=%1.1f\n", from_pos, from_pos_shift, to_pos, diff_contrib)

	return diff_contrib;
}
}
#endif
