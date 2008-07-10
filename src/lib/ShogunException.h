#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

/** Class ShogunException defines an exception which is thrown whenever an
 * error inside of shogun occurs
 */ 
class ShogunException {
	public:
		/** constructor
		 *
		 * @param str exception string
		 */
		ShogunException(const char* str);

		/** get exception string
		 *
		 * @return the exception string
		 */
		inline const char* get_exception_string() { return val; }

	private:
		/** exception string */
		char* val;
};

#endif // _SHOGUN_EXCEPTION_H_
