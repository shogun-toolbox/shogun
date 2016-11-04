import sys
import json
import argparse
import ply


# The FastParser parses input using PLY
class FastParser:
    def __init__(self, generatedFilesOutputDir=None):
        from ply import lex
        from ply import yacc

        generateFiles = not generatedFilesOutputDir is None

        if generateFiles:
            sys.path.append(generatedFilesOutputDir)

        # Build the lexer and the parser
        self.lexer = lex.lex(module=self,
                             optimize=generateFiles,
                             outputdir=generatedFilesOutputDir)
        self.parser = yacc.yacc(module=self,
                                write_tables=generateFiles,
                                outputdir=generatedFilesOutputDir,
                                debug=generateFiles)

    def parse(self, programString, filePath="Unknown"):
        """
        Parse a program string.
        The path of the program file is added as metadata to returned object
        """

        # Add trailing endline if missing
        if len(programString) and programString[-1] != "\n":
            programString += "\n"

        # Parse program and add FilePath key to object
        program = self.parser.parse(programString)
        program["FilePath"] = filePath

        return program

    # Lexer specification
    # ---------------------------------------
    tokens = (
        "INTLITERAL",
        "REALLITERAL",
        "FLOATLITERAL",
        "STRINGLITERAL",
        "CHARLITERAL",
        "BOOLLITERAL",
        "BASICTYPE",
        "SHOGUNSGTYPE",
        "PRINTKEYWORD",
        "COMMA",
        "DOT",
        "COLON",
        "ENUMKEYWORD",
        "EQUALS",
        "LPAREN",
        "RPAREN",
        "LSQUARE",
        "RSQUARE",
        "COMMENT",
        "NEWLINE",
        "IDENTIFIER"
    )

    reserved = {
        'enum': 'ENUMKEYWORD',
        'print': 'PRINTKEYWORD',
        'True': 'BOOLLITERAL',
        'False': 'BOOLLITERAL',
        'int': 'BASICTYPE',
        'bool': 'BASICTYPE',
        'float': 'BASICTYPE',
        'real': 'BASICTYPE',
        'string': 'BASICTYPE',
        'char': 'BASICTYPE',
        'BoolVector': 'SHOGUNSGTYPE',
        'CharVector': 'SHOGUNSGTYPE',
        'ByteVector': 'SHOGUNSGTYPE',
        'WordVector': 'SHOGUNSGTYPE',
        'ShortVector': 'SHOGUNSGTYPE',
        'IntVector': 'SHOGUNSGTYPE',
        'LongIntVector': 'SHOGUNSGTYPE',
        'ULongIntVector': 'SHOGUNSGTYPE',
        'ShortRealVector': 'SHOGUNSGTYPE',
        'RealVector': 'SHOGUNSGTYPE',
        'LongRealVector': 'SHOGUNSGTYPE',
        'ComplexVector': 'SHOGUNSGTYPE',
        'BoolMatrix': 'SHOGUNSGTYPE',
        'CharMatrix': 'SHOGUNSGTYPE',
        'ByteMatrix': 'SHOGUNSGTYPE',
        'WordMatrix': 'SHOGUNSGTYPE',
        'ShortMatrix': 'SHOGUNSGTYPE',
        'IntMatrix': 'SHOGUNSGTYPE',
        'LongIntMatrix': 'SHOGUNSGTYPE',
        'ULongIntMatrix': 'SHOGUNSGTYPE',
        'ShortRealMatrix': 'SHOGUNSGTYPE',
        'RealMatrix': 'SHOGUNSGTYPE',
        'LongRealMatrix': 'SHOGUNSGTYPE',
        'ComplexMatrix': 'SHOGUNSGTYPE'
    }

    t_INTLITERAL = "[0-9]+"
    t_REALLITERAL = "[0-9]+\.[0-9]+"
    t_FLOATLITERAL = "[0-9]+\.[0-9]+f"
    t_STRINGLITERAL = '"[^"\n]*"'
    t_CHARLITERAL = "'[^']{1}'"
    t_COMMA = ","
    t_DOT = "\."
    t_COLON = ":"
    t_EQUALS = "="
    t_LPAREN = "\("
    t_RPAREN = "\)"
    t_LSQUARE = "\["
    t_RSQUARE = "\]"
    t_ignore = " \t"

    def t_IDENTIFIER(self, t):
        "[a-zA-Z][a-zA-Z0-9-_]*"
        # Check for reserved words
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t

    def t_NEWLINE(self, t):
        r'\n'
        t.lexer.lineno += 1
        return t

    def t_COMMENT(self, t):
        r"\#.*\n"
        t.lexer.lineno += 1
        return t

    def t_error(self, t):
        raise TypeError("Failed to tokenize input. Unknown text on line %d '%s'" % (t.lineno, t.value,))

    # Grammar specification
    # ---------------------------------------
    def p_program(self, p):
        "program : statements"
        p[0] = {"Program": p[1]}

    def p_statements(self, p):
        """
        statements : statement statements
                   | comment statements
                   |
        """
        if len(p) > 2:
            p[0] = p[1:2]
            p[0].extend(p[2])
        else:
            p[0] = []

    def p_comment(self, p):
        "comment : COMMENT"
        # Strip leading hashtag symbol and trailing newline
        p[0] = {"Comment": p[1][1:-1],
                "__PARSER_INFO_LINE_NO": p.lineno(1)}

    def p_type(self, p):
        """
        type : basictype
             | shogunsgtype
             | objecttype
        """
        p[0] = p[1]

    def p_basicType(self, p):
        "basictype : BASICTYPE"
        p[0] = {"BasicType": p[1]}

    def p_shogunSGType(self, p):
        "shogunsgtype : SHOGUNSGTYPE"
        p[0] = {"ShogunSGType": p[1]}

    def p_objectType(self, p):
        "objecttype : IDENTIFIER"
        p[0] = {"ObjectType": p[1]}

    def p_argumentList_nonEmpty(self, p):
        """
        argumentListNonEmpty : expr
                             | expr COMMA argumentListNonEmpty
        """
        p[0] = [p[1]]
        if len(p) > 2:
            p[0].extend(p[3])

    def p_argumentList(self, p):
        """
        argumentList : argumentListNonEmpty
                     |
        """
        arguments = []
        if len(p) > 1:
            arguments = p[1]

        p[0] = {"ArgumentList": arguments}

    def p_identifier(self, p):
        "identifier : IDENTIFIER"
        p[0] = {"Identifier": p[1]}

    def p_methodCall(self, p):
        "methodCall : identifier DOT identifier LPAREN argumentList RPAREN"
        p[0] = {"MethodCall": [p[1], p[3], p[5]]}

    def p_staticCall(self, p):
        "staticCall : type COLON identifier LPAREN argumentList RPAREN"
        p[0] = {"StaticCall": [p[1], p[3], p[5]]}

    def p_indexList(self, p):
        """
        indexList : int
                  | int COMMA indexList
        """
        p[0] = [p[1]]
        if len(p) > 2:
            p[0].extend(p[3])

    def p_elementAccess(self, p):
        "elementAccess : identifier LSQUARE indexList RSQUARE"
        p[0] = {"ElementAccess": [p[1],
                                  {"IndexList": p[3]}]}

    def p_enum(self, p):
        "enum : ENUMKEYWORD identifier DOT identifier"
        p[0] = {"Enum": [p[2], p[4]]}

    def p_string(self, p):
        "string : STRINGLITERAL"
        # Strip leading and trailing quotes
        p[0] = {"StringLiteral": p[1][1:-1]}
    
    def p_char(self, p):
        "char : CHARLITERAL"
        # Strip leading and trailing quotes
        p[0] = {"CharLiteral": p[1][1:-1]}
    
    def p_bool(self, p):
        "bool : BOOLLITERAL"
        p[0] = {"BoolLiteral": p[1]}

    def p_int(self, p):
        "int : INTLITERAL"
        p[0] = {"IntLiteral": p[1]}

    def p_real(self, p):
        "real : REALLITERAL"
        p[0] = {"RealLiteral": p[1]}

    def p_float(self, p):
        "float : FLOATLITERAL"
        p[0] = {"FloatLiteral": p[1][:-1]}

    def p_expr(self, p):
        """
        expr : enum
             | methodCall
             | staticCall
             | elementAccess
             | string
             | char
             | bool
             | int
             | real
             | float
             | identifier
        """
        p[0] = {"Expr": p[1]}

    def p_assignment(self, p):
        """
        assignment : identifier EQUALS expr
                   | elementAccess EQUALS expr
        """
        p[0] = {"Assign": [p[1], p[3]]}

    def p_initialisation(self, p):
        """
        initialisation : type identifier EQUALS expr
                       | type identifier LPAREN argumentList RPAREN
        """
        p[0] = {"Init": [p[1], p[2], p[4]]}

    def p_output(self, p):
        "output : PRINTKEYWORD expr"
        p[0] = {"Print": p[2]}

    def p_statement(self, p):
        """
        statement : initialisation NEWLINE
                  | assignment NEWLINE
                  | expr NEWLINE
                  | output NEWLINE
                  | NEWLINE
        """
        p[0] = {"Statement": p[1],
                "__PARSER_INFO_LINE_NO": p.lineno(-1)}

    # Error rule for syntax errors
    def p_error(self, p):
        if p:
            print("Syntax error in input: " +
                  str(p.value) + " on line " + str(p.lineno))
        else:
            print("Reached end of file without completing parse")


def parse(programString, filePath, generatedFilesOutputDir=None):
    parser = FastParser(generatedFilesOutputDir)

    # Parse input
    return parser.parse(programString, filePath)

if __name__ == "__main__":
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretty", action="store_true", help="If specified, output is pretty printed")
    argparser.add_argument("path", nargs='?', help="Path to input file. If not specified input is read from stdin")
    argparser.add_argument("--parser_files_dir", nargs='?', help='Path to directory where generated parser and lexer files should be stored.')
    args = argparser.parse_args()

    programString = ""
    filePath = ""

    # Read from specified file or, if not specified, from stdin
    if args.path:
        with open(args.path, 'r') as file:
            programString = file.read()
        filePath = args.path
    else:
        programString = sys.stdin.read()
        filePath = "sys.stdin"

    indentWidth = 2 if args.pretty > 0 else None

    # Parse input and print json output
    program = parse(programString, filePath, args.parser_files_dir)

    print(json.dumps(program, indent=indentWidth))
