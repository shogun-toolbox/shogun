import sys
import ast
import pyparsing as pp
import json
# Memoization speed boost
pp.ParserElement.enablePackrat()
pp.ParserElement.setDefaultWhitespaceChars(' \t\r')

def grammar():
    # Identifiers are letters followed by letters, numbers, _, and -
    identifier = pp.Word(initChars=pp.alphas, bodyChars=pp.alphanums+'_-')
    # Numerals are integers or decimal number with no leading zeros
    numeral = pp.Regex('([1-9][0-9]*(\.[0-9]*)?)|0\.[0-9]*')
    # String literals are enclosed by double quotes
    stringLiteral = pp.QuotedString('"')
    boolLiteral = pp.Literal('True') ^ pp.Literal('False')
    
    basicType = pp.Literal('int') ^ pp.Literal('bool') ^ pp.Literal('float') ^ pp.Literal('string') ^ pp.Literal('RealVector')
    # Shogun types are retrieved from the type list
    objectType = shogunType()
    type = objectType ^ basicType

    expr = pp.Forward()
    argumentList = pp.Forward()
    argumentList << (expr ^ (expr + ',' + argumentList))
    methodCall = identifier + '.' + identifier + '(' + (argumentList ^ pp.empty) + ')'
    enum = 'enum' + identifier
    expr << (enum ^ stringLiteral ^ boolLiteral ^ numeral ^ methodCall ^ identifier)
    
    assignment = identifier + '=' + expr
    # Initialisation is done by passing arguments to the class constructor
    # or by copying an expression to the variable
    initialisation = type + identifier + (('(' + (argumentList ^ pp.empty) + ')') ^ ('=' + expr))
    output = 'print' + expr

    statement = pp.Optional(initialisation ^ assignment ^ expr ^ output) + pp.lineEnd

    grammar = pp.ZeroOrMore(statement)

    # Connect grammar to ast data structure
    grammar.setParseAction(ast.Program)
    statement.setParseAction(ast.Statement)
    initialisation.setParseAction(ast.Init)
    assignment.setParseAction(ast.Assign)
    output.setParseAction(ast.Print)
    expr.setParseAction(ast.Expr)
    enum.setParseAction(ast.Enum)
    methodCall.setParseAction(ast.MethodCall)
    numeral.setParseAction(ast.NumberLiteral)
    stringLiteral.setParseAction(ast.StringLiteral)
    boolLiteral.setParseAction(ast.BoolLiteral)
    basicType.setParseAction(ast.BasicType)
    objectType.setParseAction(ast.ObjectType)
    identifier.setParseAction(ast.Identifier)
    argumentList.setParseAction(ast.ArgumentList)

    return grammar

def shogunType():
    type = pp.Forward()
    with open("types/typelist", "r") as typelist:
        for line in typelist:
            # Make sure the line is not an empty string
            if line.replace('\n', '') != '':
                type = type ^ pp.Literal(line.replace('\n', ''))

    return type

if __name__ == "__main__":
    inputFile = sys.stdin
    if len(sys.argv) > 1:
        inputFile = sys.argv[1]

    program = grammar().parseFile(inputFile, parseAll=True)[0]
    print json.dumps(program, cls=ast.JSONEncoder, indent=2)
