class CSyntaxHighLight
{
	public:
		CSyntaxHighLight()
		{
			set_ansi_syntax_hilighting();
		}

		void set_ansi_syntax_hilighting()
		{
			command_prefix="\033[1;31m";
			command_suffix="\033[0m";
			prompt_prefix="\033[1;34";
			prompt_suffix="\033[0m";
		}

		void disable_syntax_hilighting()
		{
			command_prefix="";
			command_suffix="";
			prompt_prefix="";
			prompt_suffix="";
		}

		void set_doxygen_syntax_hilighting()
		{
			command_prefix="\b ";
			command_suffix="";
			prompt_prefix="";
			prompt_suffix="";
		}

		const char* get_command_prefix()
		{
			return command_prefix;
		}

		const char* get_command_suffix()
		{
			return command_suffix;
		}

		const char* get_prompt_prefix()
		{
			return prompt_prefix;
		}

		const char* get_prompt_suffix()
		{
			return prompt_suffix;
		}

	public:
		const char* command_prefix;
		const char* command_suffix;
		const char* prompt_prefix;
		const char* prompt_suffix;
};
