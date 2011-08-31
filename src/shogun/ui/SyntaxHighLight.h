/** @brief the syntax highlight */
class CSyntaxHighLight
{
	public:
		/** constructor */
		CSyntaxHighLight()
		{
			set_ansi_syntax_hilighting();
		}

		/** set ansi syntax hilighting */
		void set_ansi_syntax_hilighting()
		{
			command_prefix="\033[1;31m";
			command_suffix="\033[0m";
			prompt_prefix="\033[1;34";
			prompt_suffix="\033[0m";
		}

		/** disable syntax hiligthing */
		void disable_syntax_hilighting()
		{
			command_prefix="";
			command_suffix="";
			prompt_prefix="";
			prompt_suffix="";
		}

		/** set doxygen syntax hilighting */
		void set_doxygen_syntax_hilighting()
		{
			command_prefix="\b ";
			command_suffix="";
			prompt_prefix="";
			prompt_suffix="";
		}

		/** get command prefix
		 * @return command prefix
		 */
		const char* get_command_prefix()
		{
			return command_prefix;
		}

		/** get command suffic
		 * @return command suffix
		 */
		const char* get_command_suffix()
		{
			return command_suffix;
		}

		/** get prompt prefix
		 * @return prompt prefix
		 */
		const char* get_prompt_prefix()
		{
			return prompt_prefix;
		}

		/** get prompt suffix
		 * @return prompt suffix
		 */
		const char* get_prompt_suffix()
		{
			return prompt_suffix;
		}

	public:
		/** command prefix */
		const char* command_prefix;
		/** command suffix */
		const char* command_suffix;
		/** prompt prefix */
		const char* prompt_prefix;
		/** prompt suffix */
		const char* prompt_suffix;
};
