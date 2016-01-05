import sys
import json
import os
import argparse
import imp

# The SlowParer parses input using the pyparsign module
class SlowParser:
    def __init__(self):
        import pyparsing as pp
        self.pp = pp
        # Memoization speed boost
        self.pp.ParserElement.enablePackrat()
        self.pp.ParserElement.setDefaultWhitespaceChars(' \t\r')

    def parse(self, programString, filePath="Unknown"):
        # Add trailing endline if missing
        if programString[-1] != "\n":
            programString += "\n"

        program = self.grammar().parseString(programString, parseAll=True)[0]
        programAST = self.JSONEncoder().default(program)
        programAST["FilePath"] = filePath
        return programAST

    def shogunType(self):
        basepath = os.path.dirname(__file__)
        type = self.pp.Forward()

        # This is probably a temporary solution, we treat any [a-zA-Z_][a-zA-Z0-9_]* as valid shogun keyword
        type = type ^ self.pp.Word(self.pp.srange("[a-zA-Z_]"), self.pp.srange("[a-zA-Z0-9_]"))

        # with open(os.path.join(basepath,"types/typelist"), "r") as typelist:
        #     for line in typelist:
        #         # Make sure the line is not an empty string
        #         if line.replace('\n', '') != '':
        #             type = type ^ self.pp.Literal(line.replace('\n', ''))
        return type

    def grammar(self):
        # Identifiers are letters followed by letters, numbers, _, and -
        identifier = self.pp.Word(initChars=self.pp.alphas, bodyChars=self.pp.alphanums+'_-')
        # Numerals are integers or decimal number with no leading zeros
        numeral = self.pp.Regex('([1-9][0-9]*(\.[0-9]*)?)|0\.[0-9]*')
        # String literals are enclosed by double quotes
        stringLiteral = self.pp.QuotedString('"')
        boolLiteral = self.pp.Literal('True') ^ self.pp.Literal('False')

        basicType = self.pp.Literal('int') ^ self.pp.Literal('bool') ^ self.pp.Literal('float') ^ self.pp.Literal('string') ^ self.pp.Literal('RealVector')
        # Shogun types are retrieved from the type list
        objectType = self.shogunType()
        type = objectType ^ basicType

        expr = self.pp.Forward()
        nonEmptyArgumentList = self.pp.Forward()
        nonEmptyArgumentList << (expr ^ (expr + ',' + nonEmptyArgumentList))
        argumentList = nonEmptyArgumentList ^ self.pp.empty
        methodCall = identifier + '.' + identifier + '(' + argumentList + ')'
        enum = 'enum' + identifier + '.' + identifier
        expr << (enum ^ stringLiteral ^ boolLiteral ^ numeral ^ methodCall ^ identifier)

        assignment = identifier + '=' + expr
        # Initialisation is done by passing arguments to the class constructor
        # or by copying an expression to the variable
        initialisation = type + identifier + (('(' + argumentList + ')') ^ ('=' + expr))
        output = 'print' + expr

        comment = '#' + self.pp.restOfLine + self.pp.lineEnd

        statement = (self.pp.Optional(initialisation ^ assignment ^ expr ^ output) + self.pp.lineEnd)

        grammar = self.pp.ZeroOrMore(statement ^ comment)

        # Connect grammar to ast data structure
        grammar.setParseAction(self.Program)
        statement.setParseAction(self.Statement)
        initialisation.setParseAction(self.Init)
        assignment.setParseAction(self.Assign)
        output.setParseAction(self.Print)
        expr.setParseAction(self.Expr)
        comment.setParseAction(self.Comment)
        enum.setParseAction(self.Enum)
        methodCall.setParseAction(self.MethodCall)
        numeral.setParseAction(self.NumberLiteral)
        stringLiteral.setParseAction(self.StringLiteral)
        boolLiteral.setParseAction(self.BoolLiteral)
        basicType.setParseAction(self.BasicType)
        objectType.setParseAction(self.ObjectType)
        identifier.setParseAction(self.Identifier)
        argumentList.setParseAction(self.ArgumentList)

        return grammar

    # AST specification
    class BaseNode(object):
        def __init__(self, string, location, tokens):
            self.tokens = tokens
            self.string = string
            self.location = location

    class Program(BaseNode): pass
    class Statement(BaseNode): pass
    class Init(BaseNode): pass
    class Assign(BaseNode): pass
    class Print(BaseNode): pass
    class Identifier(BaseNode): pass
    class ArgumentList(BaseNode): pass
    class Expr(BaseNode): pass
    class Comment(BaseNode): pass
    class Enum(Expr): pass
    class MethodCall(Expr): pass
    class Type(BaseNode): pass # Abstract. Don't use directly
    class BasicType(Type): pass
    class ObjectType(Type): pass
    class Literal(Expr): pass # Abstract. Don't use directly
    class BoolLiteral(Literal): pass
    class NumberLiteral(Literal): pass
    class StringLiteral(Literal): pass

    class JSONEncoder(json.JSONEncoder):
        def default(self, node):
            """ Encodes the AST into objects that are JSON serializable """

            if issubclass(node.__class__, SlowParser.ArgumentList):
                # Flatten argument lists: [Expr, [Expr, [Expr]]] = [Expr, Expr, Expr]
                while len(node.tokens) > 0 and isinstance(node.tokens[-1], SlowParser.ArgumentList):
                    tail = node.tokens[-1].tokens
                    node.tokens = node.tokens[0:-1]
                    node.tokens.extend(tail)

                # Make sure empty argument lists get encoded as empty lists and not null
                if len(node.tokens) == 0:
                    node.tokens = [[]]

            if issubclass(node.__class__, SlowParser.Program):
                # Remove the EOF token if it got parsed as an individual statement
                if len(node.tokens[-1].tokens) == 0:
                    node.tokens = node.tokens[0:-1]

            if issubclass(node.__class__, SlowParser.Comment):
                # Remove beginning # from comments
                node.tokens = [node.tokens[1]]

            if issubclass(node.__class__, SlowParser.BaseNode):
                if len(node.tokens) > 1:
                    # If tokens is a list we only consider instances of BaseClass subclasses
                    objectList = [self.default(element) for element in node.tokens if issubclass(element.__class__, SlowParser.BaseNode)]
                    if len(objectList) == 1:
                        objectList = objectList[0]

                    return {node.__class__.__name__: objectList}
                else:
                    tokens = None
                    # Handle ArgumentLists of length 0 or 1
                    if isinstance(node, SlowParser.ArgumentList):
                        tokens = self.default(node.tokens[0])
                        # Ensure that tokens stays a list when it has a single object
                        if len(tokens) > 0:
                            tokens = [tokens]
                    elif len(node.tokens) > 0:
                        tokens = self.default(node.tokens[0])

                    # For non-lists, we pass on the token no matter its type
                    return {node.__class__.__name__:tokens}

            return node

# The FastParser parses input using PLY
class FastParser:
    def __init__(self):
        from ply import lex
        from ply import yacc

        # Add all shogun types to reserved identifiers
        # (this is commented out as we allow any identifier to be shogun one)
        #self.addShogunTypes(self.reserved)

        # Build the lexer and the parser
        self.lexer = lex.lex(module=self,optimize=1)
        self.parser = yacc.yacc(module=self)

    def parse(self, programString, filePath="Unknown"):
        """
        Parse a program string.
        The path of the program file is added as metadata to returned object
        """

        # Add trailing endline if missing
        if programString[-1] != "\n":
            programString += "\n"

        # Parse program and add FilePath key to object
        program = self.parser.parse(programString)
        program["FilePath"] = filePath

        return program

    def addShogunTypes(self, dictionary):
        "Reads shogun types from types/typelist and adds them to dictionary"

        basepath = os.path.dirname(__file__)
        with open(os.path.join(basepath,"types/typelist"), "r") as typelist:
            for line in typelist:
                # Make sure the line is not an empty string
                if line.replace('\n', '') != '':
                    dictionary[line.replace('\n', '')] = "SHOGUNTYPE"

    # Lexer specification
    # ---------------------------------------
    tokens = (
        "NUMERAL",
        "STRINGLITERAL",
        "BOOLLITERAL",
        "BASICTYPE",
        #"SHOGUNTYPE",
        "PRINTKEYWORD",
        "COMMA",
        "DOT",
        "ENUMKEYWORD",
        "EQUALS",
        "LPAREN",
        "RPAREN",
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
        'float':  'BASICTYPE',
        'string': 'BASICTYPE',
        'RealVector': 'BASICTYPE'
    }

    t_NUMERAL = "([1-9][0-9]*(\.[0-9]+)?)|0\.[0-9]+"
    t_STRINGLITERAL = '"[^"\n]*"'
    t_COMMA = ","
    t_DOT = "\."
    t_EQUALS = "="
    t_LPAREN = "\("
    t_RPAREN = "\)"
    t_COMMENT = r"\#.*\n"
    t_ignore  = " \t"

    def t_IDENTIFIER(self, t):
        "[a-zA-Z][a-zA-Z0-9-_]*"
        t.type = self.reserved.get(t.value,'IDENTIFIER')    # Check for reserved words
        return t

    def t_NEWLINE(self, t):
        r'\n'
        t.lexer.lineno += len(t.value)
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
        p[0] = {"Comment": p[1][1:-1]}

    def p_type(self, p):
        """
        type : basictype
             | objecttype
        """
        p[0] = p[1]

    def p_basicType(self, p):
        "basictype : BASICTYPE"
        p[0] = {"BasicType": p[1]}

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

    def p_enum(self, p):
        "enum : ENUMKEYWORD identifier DOT identifier"
        p[0] = {"Enum": [p[2], p[4]]}

    def p_string(self, p):
        "string : STRINGLITERAL"
        # Strip leading and trailing quotes
        p[0] = {"StringLiteral": p[1][1:-1]}

    def p_bool(self, p):
        "bool : BOOLLITERAL"
        p[0] = {"BoolLiteral": p[1]}

    def p_numeral(self, p):
        "numeral : NUMERAL"
        p[0] = {"NumberLiteral": p[1]}

    def p_expr(self, p):
        """
        expr : enum
             | methodCall
             | string
             | bool
             | numeral
             | identifier
        """
        p[0] = {"Expr": p[1]}

    def p_assignment(self, p):
        "assignment : identifier EQUALS expr"
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
        p[0] = {"Statement": p[1]}

    # Error rule for syntax errors
    def p_error(self, p):
        if p:
            print "Syntax error in input: " + str(p.value) + " on line " + str(p.lineno)
        else:
            print "Reached end of file without completing parse"


def parse(programString, filePath):
    # Check if PLY is installed
    try:
        imp.find_module('ply')
        plyFound = True
    except ImportError:
        plyFound = False

    # Use the fast parser if PLY was found
    parser = None
    if plyFound:
        parser = FastParser()
    else:
        parser = SlowParser()

    # Parse input
    return parser.parse(programString, filePath)

if __name__ == "__main__":
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretty", action="store_true", help="If specified, output is pretty printed")
    argparser.add_argument("path", nargs='?', help="Path to input file. If not specified input is read from stdin")
    args = argparser.parse_args()

    programString = ""
    filePath = ""

    # Read from specified file or, if not specified, from stdin
    if args.path :
        with open(args.path, 'r') as file:
            programString = file.read()
        filePath = args.path
    else:
        programString = sys.stdin.read()
        filePath = "sys.stdin"

    indentWidth = 2 if args.pretty > 0 else None

    # Parse input and print json output
    program = parse(programString, filePath)

    print json.dumps(program, indent=indentWidth)
