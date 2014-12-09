import json

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

        if issubclass(node.__class__, ArgumentList):
            # Flatten argument lists: [Expr, [Expr, [Expr]]] = [Expr, Expr, Expr]
            while isinstance(node.tokens[-1], ArgumentList):
                tail = node.tokens[-1].tokens
                node.tokens = node.tokens[0:-1]
                node.tokens.extend(tail)

        if issubclass(node.__class__, Program):
            # Remove the EOF token if it got parsed as an individual statement
            if len(node.tokens[-1].tokens) == 0:
                node.tokens = node.tokens[0:-1]

        if issubclass(node.__class__, BaseNode):
            if len(node.tokens) > 1:
                # If tokens is a list we only consider instances of BaseClass subclasses
                objectList = [self.default(element) for element in node.tokens if issubclass(element.__class__, BaseNode)]
                if len(objectList) == 1:
                    objectList = objectList[0]

                return {node.__class__.__name__: objectList}
            else:
                tokens = None
                if len(node.tokens) > 0:
                    tokens = self.default(node.tokens[0])

                # For non-lists, we pass on the token no matter its type
                return {node.__class__.__name__:tokens}

        return node