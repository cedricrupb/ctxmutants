import libcst as cst

from libcst import MaybeSentinel, Param, ParamStar, ParamSlash, If, IsNot, NotIn

from contextlib import contextmanager

from .token import TokenTypes, TokenizationState


class CSTFuncVisitor(cst.CSTVisitor):

    def __init__(self):
        self.func_defs = {}

    def visit_FunctionDef(self, node):

        state = TokenizationState()
        NodeTokenizer(state)(node)

        self.func_defs[node.name.value] = {
            "tokens": state.tokens,
            "types" : state.type_ids
        }

        return False


class NodeTokenizer:

    def __init__(self, state):
        self.state = state

    def __call__(self, node): return self.tokenize(node)

    def tokenize(self, node):
        target_func = getattr(self, f"tokenize_{type(node).__name__}", None)
        if target_func is not None:
            target_func(node)
        else:
            raise ValueError("Cannot handle node type: %s" % type(node).__name__)

    # Utils ----------------------------------------------

    def tok(self, expr):
        if hasattr(expr, "_get_token"):
            return expr._get_token()
        if isinstance(expr, IsNot):
            return "is not"
        if isinstance(expr, NotIn):
            return "not in"
        raise ValueError("Cannot tokenize: %s" % str(expr))
    
    @contextmanager
    def parenthesize(self, node):
        for lpar in node.lpar: self.tokenize(lpar)

        yield

        for rpar in node.rpar: self.tokenize(rpar)

    @contextmanager
    def brackets(self, node):
        self.tokenize(node.lbracket)

        yield

        self.tokenize(node.rbracket)

    @contextmanager
    def braces(self, node):
        self.tokenize(node.lbrace)

        yield

        self.tokenize(node.rbrace)


    def tokenize_MaybeSentinel(self, node):
        pass

    def tokenize_Comma(self, node):
        self.state.add(",")

    def tokenize_None(self, node):
        pass

    def tokenize_NoneType(self, node):
        pass

    def tokenize_Semicolon(self, node):
        self.state.add(";")

    # Brackets ----------------------------------------------------------------
    def tokenize_LeftParen(self, node):
        self.state.add("(")

    def tokenize_RightParen(self, node):
        self.state.add(")")

    def tokenize_LeftSquareBracket(self, node):
        self.state.add("[")

    def tokenize_RightSquareBracket(self, node):
        self.state.add("]")

    def tokenize_LeftCurlyBrace(self, node):
        self.state.add("{")

    def tokenize_RightCurlyBrace(self, node):
        self.state.add("}")
    
    def tokenize_AssignEqual(self, node):
        self.state.add("=")

    # Name and Attributes --------------------------------

    def tokenize_Name(self, node):
        state   = self.state
        name    = node.value
        context = state.context

        # Assume that all names follow convention
        # Therefore: USE => USE_TYPE if name is camel case
        if context == "USE":
            if name[0].isupper() and name[1].islower():
                context = "USE_TYPE"

        ctx2id  = {"FUNC_DEF": TokenTypes.DEF_FUNC, 
                    "Param": TokenTypes.DEF_VAR, 
                    "Annotation": TokenTypes.USE_TYPE,
                    "FUNC_CALL": TokenTypes.CALL_FUNC,
                    "USE": TokenTypes.USE_VAR,
                    "DEF": TokenTypes.DEF_VAR,
                    "ATTR": TokenTypes.ATTR,
                    "USE_TYPE": TokenTypes.USE_TYPE}
        type_id = ctx2id[context]

        with self.parenthesize(node):
            state.add(name, type_id) 
        

    def tokenize_Attribute(self, node):
        state = self.state
        with self.parenthesize(node):

            with state.new_ctx("USE"):
                self.tokenize(node.value)

            state.add(self.tok(node.dot))

            if state.context == "FUNC_CALL": # It is a method callFunc
                self.tokenize(node.attr)
            else:
                with state.new_ctx("ATTR"):
                    self.tokenize(node.attr)


    # Operations ------------------

    def tokenize_UnaryOperation(self, node):
        state = self.state
        with self.parenthesize(node):
            state.add(
                self.tok(node.operator), TokenTypes.UOP
            )

            with state.new_ctx("USE"):
                self.tokenize(node.expression)

    def tokenize_BinaryOperation(self, node):
        state = self.state
        with self.parenthesize(node):
            with state.new_ctx("USE"):
                self.tokenize(node.left)
            state.add(
                self.tok(node.operator), TokenTypes.BOP
            )
            with state.new_ctx("USE"):
                self.tokenize(node.right)

    def tokenize_BooleanOperation(self, node):
        self.tokenize_BinaryOperation(node)

    def tokenize_Comparison(self, node):
        with self.parenthesize(node):
            with self.state.new_ctx("USE"):
                self.tokenize(node.left)

                for comp in node.comparisons:
                    self.tokenize(comp)

    def tokenize_ComparisonTarget(self, node):
        state = self.state
        state.add(
            self.tok(node.operator), TokenTypes.BOP
        )

        self.tokenize(node.comparator)

    # Control Flow --------------------------------

    def tokenize_Asynchronous(self, node):
        self.state.add("async", TokenTypes.KEYWORDS)
        
    def tokenize_Await(self, node):
        self.state.add("await", TokenTypes.KEYWORDS)

        with self.state.new_ctx("USE"):
            self.tokenize(node.expression)

    def tokenize_Yield(self, node):
        self.state.add("yield", TokenTypes.KEYWORDS)
        self.tokenize(node.value)

    def tokenize_From(self, node):
        self.state.add("from", TokenTypes.KEYWORDS)
        self.tokenize(node.item)

    def tokenize_IfExp(self, node):
        state = self.state
        with self.parenthesize(node):
            self.tokenize(node.body)
            state.add("if", TokenTypes.KEYWORDS)
            self.tokenize(node.test)
            state.add("else", TokenTypes.KEYWORDS)
            self.tokenize(node.orelse)

    # Lambda and Fn -------------------------

    def tokenize_Lambda(self, node):
        state = self.state

        state.add("lambda", TokenTypes.KEYWORDS)
        self.tokenize(node.params)
        self.tokenize(node.colon)
        self.tokenize(node.body)


    def tokenize_Call(self, node):
        with self.parenthesize(node):
            state = self.state

            with state.new_ctx("FUNC_CALL"):
                self.tokenize(node.func)
            
            state.add("(")

            lastarg = len(node.args) - 1
            for i, arg in enumerate(node.args):
                self.tokenize(arg)
                if i < lastarg: state.add(",")

            state.add(")")


    def tokenize_Arg(self, node):
        state = self.state

        with state.new_ctx("USE"):
            if len(node.star) > 0: state.add(node.star)
            keyword = node.keyword
            if keyword is not None:
                self.tokenize(keyword)
            equal = node.equal
            if equal is MaybeSentinel.DEFAULT and node.keyword is not None:
                state.add("=")
            else:
                self.tokenize(equal)
            self.tokenize(node.value)
            self.tokenize(node.comma)


    # Literals ------------------------------

    def tokenize_Ellipsis(self, node):
        with self.parenthesize(node):
            self.state.add("...")

    def tokenize_Integer(self, node):
        with self.parenthesize(node):
            self.state.add(node.value, TokenTypes.LIT_NUMBER)
    
    def tokenize_Float(self, node):
        with self.parenthesize(node):
            self.state.add(node.value, TokenTypes.LIT_NUMBER)

    def tokenize_Imaginary(self, node):
        with self.parenthesize(node):
            self.state.add(node.value, TokenTypes.LIT_NUMBER)

    def tokenize_SimpleString(self, node):
        with self.parenthesize(node):
            self.state.add(node.value, TokenTypes.LIT_STRING)

    def tokenize_ConcatenatedString(self, node):
        with self.parenthesize(node):
            self.tokenize(node.left)
            self.tokenize(node.right)

    def tokenize_FormattedString(self, node):
        with self.parenthesize(node):
            self.state.add(node.start)
            for part in node.parts:
                self.tokenize(part)
            self.state.add(node.end)

    def tokenize_FormattedStringText(self, node):
        self.state.add(node.value)

    def tokenize_FormattedStringExpression(self, node):
        state = self.state

        state.add("{")

        self.tokenize(node.expression)

        equal = node.equal
        if equal is not None:
            self.tokenize(equal)

        conversion = node.conversion
        if conversion is not None:
            state.add("!")
            state.add(conversion)
        format_spec = node.format_spec
        if format_spec is not None:
            state.add(":")
            for spec in format_spec:
                self.tokenize(spec)

        state.add("}")

    # Collections  -------------------------------------

    def tokenize_Tuple(self, node):
        state = self.state
        with self.parenthesize(node):
            elements = node.elements
            if len(elements) == 1:
                self.tokenize(elements[0])
            else:
                for el in elements:
                    self.tokenize(el)
    
    def tokenize_List(self, node):

        with self.parenthesize(node), self.brackets(node):
            elements = node.elements
            for el in elements:
                self.tokenize(el)
    

    def tokenize_Set(self, node):

        with self.parenthesize(node), self.braces(node):
            elements = node.elements
            for el in elements:
                self.tokenize(el)

    def tokenize_Dict(self, node):

        with self.parenthesize(node), self.braces(node):
            elements = node.elements
            for el in elements:
                self.tokenize(el)

    # Collection elements -----------------------------

    def tokenize_Element(self, node):
        state = self.state
        
        self.tokenize(node.value)
        
        comma = node.comma
        if comma is MaybeSentinel.DEFAULT:
            state.add(",")
        else:
            self.tokenize(comma)

    def tokenize_StarredElement(self, node):
        state = self.state

        state.add("*")
        self.tokenize(node.value)
        
        comma = node.comma
        if comma is MaybeSentinel.DEFAULT:
            state.add(",")
        else:
            self.tokenize(comma)

    def tokenize_DictElement(self, node):
        state = self.state

        self.tokenize(node.key)
        state.add(":")
        self.tokenize(node.value)
        
        comma = node.comma
        if comma is MaybeSentinel.DEFAULT:
            state.add(",")
        else:
            self.tokenize(comma)

    def tokenize_StarredDictElement(self, node):
        state = self.state

        state.add("**")
        self.tokenize(node.value)
        
        comma = node.comma
        if comma is MaybeSentinel.DEFAULT:
            state.add(",")
        else:
            self.tokenize(comma)

    # Comprehension ------------------------------------

    def tokenize_GeneratorExpr(self, node):
        with self.parenthesize(node):
            self.tokenize(node.elt)
            self.tokenize(node.for_in)
    
    def tokenize_ListComp(self, node):
        with self.parenthesize(node), self.brackets(node):
            self.tokenize(node.elt)
            self.tokenize(node.for_in)

    def tokenize_SetComp(self, node):
        with self.parenthesize(node), self.braces(node):
            self.tokenize(node.elt)
            self.tokenize(node.for_in)

    def tokenize_DictComp(self, node):
        with self.parenthesize(node), self.braces(node):
            self.tokenize(node.key)
            self.state.add(":")
            self.tokenize(node.value)
            self.tokenize(node.for_in)

    def tokenize_CompFor(self, node):
        state = self.state
        
        asynchronous = node.asynchronous
        if asynchronous is not None:
            self.tokenize(asynchronous)

        state.add("for", TokenTypes.KEYWORDS)

        self.tokenize(node.target)

        state.add("in", TokenTypes.KEYWORDS)
    
        self.tokenize(node.iter)
        ifs = node.ifs
        for if_clause in ifs:
            self.tokenize(if_clause)
        inner_for_in = node.inner_for_in
        if inner_for_in is not None:
            self.tokenize(inner_for_in)

    def tokenize_CompIf(self, node):
        state = self.state
        state.add("if", TokenTypes.KEYWORDS)
        self.tokenize(node.test)

    def tokenize_NamedExpr(self, node):
        with self.parenthesize(node):
            with self.state.new_ctx("DEF"):
                self.tokenize(node.target)
            self.state.add(":=")
            with self.state.new_ctx("USE"):
                self.tokenize(node.value)

    # Subscript ------------------------------
    
    def tokenize_Subscript(self, node):
        with self.parenthesize(node):
            self.tokenize(node.value)
            self.tokenize(node.lbracket)
            for slice in node.slice:
                self.tokenize(slice)
            
            self.tokenize(node.rbracket)
    
    def tokenize_Index(self, node):
        with self.state.new_ctx("USE"):
            self.tokenize(node.value)

    def tokenize_Slice(self, node):
        state = self.state

        lower = node.lower
        if lower is not None:
            self.tokenize(lower)
        
        self.tokenize(node.first_colon)

        upper = node.upper
        if upper is not None:
            self.tokenize(upper)

        second_colon = node.second_colon
        if second_colon is MaybeSentinel.DEFAULT and self.step is not None:
            state.add(":")
        else:
            self.tokenize(second_colon)
            
        step = self.step
        if step is not None:
            self.tokenize(step)

    def tokenize_SubscriptElement(self, node):
        self.tokenize(node.slice)

        comma = node.comma
        if comma is MaybeSentinel.DEFAULT :
            self.state.add(",")
        else:
            self.tokenize(comma)

    # Statements -----------------------------

    def tokenize_AnnAssign(self, node):
        state = self.state

        with state.new_ctx("DEF"):
            self.tokenize(node.target)
        
        state.add(":")
        self.tokenize(node.annotation)

        self.tokenize(node.equal)

        if node.value is not None:
            with state.new_ctx("USE"):
                self.tokenize(node.value)
        self.tokenize(node.semicolon)

    def tokenize_Assert(self, node):
        state = self.state

        state.add("assert", TokenTypes.KEYWORDS)
        self.tokenize(node.test)

        msg = node.msg
        if msg is not None:
            state.add(",")
            self.tokenize(msg)
        
        self.tokenize(node.semicolon)

    def tokenize_AssignTarget(self, node):
        self.tokenize(node.target)
        self.state.add("=")

    def tokenize_Assign(self, node):
        state = self.state

        with state.new_ctx("DEF"):
            for target in node.targets: 
                self.tokenize(target)
        
        with state.new_ctx("USE"):
            self.tokenize(node.value)

        self.tokenize(node.semicolon)

    def tokenize_AugAssign(self, node):
        state = self.state

        with state.new_ctx("DEF"):
            self.tokenize(node.target)
        
        state.add(self.tok(node.operator), TokenTypes.BOP)

        with state.new_ctx("USE"):
            self.tokenize(node.value)
        self.tokenize(node.semicolon)

    def tokenize_Break(self, node):
        self.state.add("break", TokenTypes.KEYWORDS)
        self.tokenize(node.semicolon)

    def tokenize_Continue(self, node):
        self.state.add("continue", TokenTypes.KEYWORDS)
        self.tokenize(node.semicolon)

    def tokenize_Del(self, node):
        self.state.add("del", TokenTypes.KEYWORDS)
        self.tokenize(node.target)
        self.tokenize(node.semicolon)

    def tokenize_Expr(self, node):
        self.tokenize(node.value)
        self.tokenize(node.semicolon)

    def tokenize_Global(self, node):
        state = self.state
        state.add("global", TokenTypes.KEYWORDS)

        lastname = len(node.names) - 1
        for i, name in enumerate(node.names):
            with state.new_ctx("DEF"):
                self.tokenize(name)
            if i < lastname: state.add(",")
        self.tokenize(node.semicolon)

    def tokenize_Nonlocal(self, node):
        state = self.state
        state.add("nonlocal", TokenTypes.KEYWORDS)

        lastname = len(node.names) - 1
        for i, name in enumerate(node.names):
            with state.new_ctx("DEF"):
                self.tokenize(name)
            if i < lastname: state.add(",")
        self.tokenize(node.semicolon)
            
    
    def tokenize_Pass(self, node):
        self.state.add("pass", TokenTypes.KEYWORDS)
        self.tokenize(node.semicolon)

    def tokenize_Raise(self, node):
        state = self.state
        exc = node.exc
        cause = node.cause
        
        state.add("raise", TokenTypes.KEYWORDS)

        if exc is not None:
            self.tokenize(exc)
        
        if cause is not None:
            self.tokenize(cause)
        self.tokenize(node.semicolon)


    def tokenize_Return(self, node):
        state = self.state

        state.add("return", TokenTypes.KEYWORDS)

        with state.new_ctx("USE"):
            self.tokenize(node.value)

        self.tokenize(node.semicolon)

    def tokenize_NameItem(self, node):
        self.tokenize(node.name)
        self.tokenize(node.comma)


    # Compound statements ----------------------

    def tokenize_For(self, node):
        asynchronous = node.asynchronous
        if asynchronous is not None:
            self.tokenize(asynchronous)
        
        state = self.state
        state.add("for", TokenTypes.KEYWORDS)

        with state.new_ctx("DEF"):
            self.tokenize(node.target)
        
        state.add("in", TokenTypes.KEYWORDS)

        with state.new_ctx("USE"):
            self.tokenize(node.iter)
        
        state.add(":")

        self.tokenize(node.body)

        if node.orelse is not None:
            self.tokenize(node.orelse)


    def tokenize_FunctionDef(self, node):
        state = self.state

        for decorator in node.decorators:
            self.tokenize(decorator)

        if node.asynchronous is not None:
             self.tokenize(node.asynchronous)

        with state.new_ctx("FUNC_DEF"):
            state.add("def", TokenTypes.KEYWORDS)
            self.tokenize(node.name)
            state.add("(")
            self.tokenize(node.params)
            state.add(")")

            if node.returns is not None:
                state.add("->")
                self.tokenize(node.returns)
            
            state.add(":")
            state.newline()

        self.tokenize(node.body)

    def tokenize_If(self, node):
        state = self.state

        state.add("elif" if state.context == "ELIF" else "if", TokenTypes.KEYWORDS)

        with state.new_ctx("USE"):
            self.tokenize(node.test)

        state.add(":")
        self.tokenize(node.body)
        
        orelse = node.orelse
        if orelse is not None:
            if isinstance(orelse, If):
                with state.new_ctx("ELIF"):
                    self.tokenize(orelse)
            else:
                self.tokenize(orelse)


    def tokenize_Else(self, node):
        self.state.add("else", TokenTypes.KEYWORDS)
        self.state.add(":")
        self.tokenize(node.body)

    def tokenize_Try(self, node):
        self.state.add("try", TokenTypes.KEYWORDS)
        self.state.add(":")

        self.tokenize(node.body)

        for handler in node.handlers: 
            self.tokenize(handler)

        if node.orelse is not None:
            self.tokenize(node.orelse)
        
        if node.finalbody is not None:
            self.tokenize(node.finalbody)


    def tokenize_While(self, node):
        state = self.state

        state.add("while", TokenTypes.KEYWORDS)
        
        with state.new_ctx("USE"):
            self.tokenize(node.test)
        
        state.add(":")

        self.tokenize(node.body)
        if node.orelse:
            self.tokenize(node.orelse)
    
    def tokenize_With(self, node):
        state = self.state

        asynchronous = node.asynchronous
        if asynchronous is not None:
            self.tokenize(asynchronous)
        
        state.add("with", TokenTypes.KEYWORDS)

        last_item = len(node.items) - 1
        for i, item in enumerate(node.items):
            self.tokenize(item)
            if i < last_item: state.add(",")
        
        state.add(":")
        self.tokenize(node.body)

    def tokenize_WithItem(self, node):
        self.tokenize(node.item)

        asname = node.asname
        if asname is not None:
            self.tokenize(asname)

        if node.comma is not None:
            self.tokenize(node.comma)

    # Helpers ----------------------------

    def tokenize_Annotation(self, node):
        with self.state.new_ctx("Annotation"):
            self.tokenize(node.annotation)

    def tokenize_AsName(self, node):
        self.state.add("as", TokenTypes.KEYWORDS)

        with self.state.new_ctx("DEF"):
            self.tokenize(node.name)

    def tokenize_ExceptHandler(self, node):
        self.state.add("except", TokenTypes.KEYWORDS)

        typenode = node.type
        if typenode is not None:
            with self.state.new_ctx("USE_TYPE"):
                self.tokenize(typenode)

        namenode = node.name
        if namenode is not None:
            with self.state.new_ctx("DEF_VAR"):
                self.tokenize(namenode)

        self.state.add(":")
        self.tokenize(node.body)

    def tokenize_Finally(self, node):
        self.state.add("finally", TokenTypes.KEYWORDS)
        self.state.add(":")
        self.tokenize(node.body)

    def tokenize_Decorator(self, node):
        self.state.add("@")

        with self.state.new_ctx("FUNC_CALL"):
            self.tokenize(node.decorator)


    def tokenize_Parameters(self, node):
        state = self.state

        star_arg = node.star_arg
        if isinstance(star_arg, MaybeSentinel):
            starincluded = len(node.kwonly_params) > 0
        elif isinstance(star_arg, (Param, ParamStar)):
            starincluded = True
        else:
            starincluded = False
        # Render out the positional-only params first. They will always have trailing
        # commas because in order to have positional-only params, there must be a
        # slash afterwards.
        for i, param in enumerate(node.posonly_params):
            self.tokenize(param)
            state.add(",")
        # Render out the positional-only indicator if necessary.
        more_values = (
            starincluded
            or len(node.params) > 0
            or len(node.kwonly_params) > 0
            or node.star_kwarg is not None
        )
        posonly_ind = node.posonly_ind
        if isinstance(posonly_ind, ParamSlash):
            # Its explicitly included, so render the version we have here which
            # might have spacing applied to its comma.
            self.tokenize(posonly_ind)
        elif len(node.posonly_params) > 0:
            state.add("/")
            if more_values: state.add(",")
            
        # Render out the params next, computing necessary trailing commas.
        lastparam = len(node.params) - 1
        more_values = (
            starincluded or len(node.kwonly_params) > 0 or node.star_kwarg is not None
        )
        for i, param in enumerate(node.params):
            self.tokenize(param)
            if (i < lastparam or more_values): state.add(",")

        # Render out optional star sentinel if its explicitly included or
        # if we are inferring it from kwonly_params. Otherwise, render out the
        # optional star_arg.
        if isinstance(star_arg, MaybeSentinel):
            if starincluded:
                state.add("*")
                state.add(",")
        else:
            self.tokenize(star_arg)

        # Render out the kwonly_args next, computing necessary trailing commas.
        lastparam = len(node.kwonly_params) - 1
        more_values = node.star_kwarg is not None
        for i, param in enumerate(node.kwonly_params):
            self.tokenize(param)
            if (i < lastparam or more_values): state.add(",")

        # Finally, render out any optional star_kwarg
        star_kwarg = node.star_kwarg
        if star_kwarg is not None:
            self.tokenize(star_kwarg)


    def tokenize_Param(self, node):
        state = self.state

        if isinstance(node.star, str) and len(node.star) > 0: state.add(node.star)

        with state.new_ctx("Param"):
            self.tokenize(node.name)

        annotation = node.annotation
        if annotation is not None:
            state.add(":")
            self.tokenize(annotation)

        equal = node.equal
        if equal is MaybeSentinel.DEFAULT and node.default is not None:
            state.add("=")
        elif equal is not None:
            self.tokenize(equal)

        default = node.default
        if default is not None:
            self.tokenize(default)

    def tokenize_ParamStar(self, node):
        state = self.state
        state.add("*")
        self.tokenize(node.comma)
    
    def tokenize_ParamSlash(self, node):
        state = self.state
        state.add("/")
        self.tokenize(node.comma)

    def tokenize_IndentedBlock(self, node):
        state = self.state

        state.indent()
        for statement in node.body:
            self.tokenize(statement)

        state.dedent()

    def tokenize_SimpleStatementLine(self, node):
        for statement in node.body:
            self.tokenize(statement)
        self.state.newline()

    def tokenize_SimpleStatementSuite(self, node):
        for statement in node.body:
            self.tokenize(statement)
        self.state.newline()


def func_tokenize(code):
    tree = cst.parse_module(code)
    visitor = CSTFuncVisitor()
    tree.visit(visitor)
    return visitor.func_defs