import javalang

from contextlib import contextmanager

from .token import TokenTypes, TokenizationState


class NodeTokenizer:

    def __init__(self, state):
        self.state = state
        self.scope = set()

    def __call__(self, node): return self.tokenize(node)

    def tokenize(self, node):
        target_func = getattr(self, f"tokenize_{type(node).__name__}", None)
        if target_func is not None:
            target_func(node)
        else:
            raise ValueError("Cannot handle node type: %s" % type(node).__name__)

    # Utils ------------------------------------

    def tokenize_List(self, L, divider=","):

        lastelem = len(L) - 1
        for i, e in enumerate(L):
            self.tokenize(e)
            if i != lastelem: self.state.add(divider) 

    @contextmanager
    def brackets(self, lpar = "(", rpar = ")"):
        self.state.add(lpar)
        yield
        self.state.add(rpar)

    @contextmanager
    def pre_post(self, node):

        if node.prefix_operators is not None:
            for op in node.prefix_operators:
                self.state.add(op, TokenTypes.UOP)
        
        yield

        if node.postfix_operators is not None:
            for op in node.postfix_operators:
                self.state.add(op, TokenTypes.UOP)


    def _identify_name(self, name):
        if name in self.scope: return TokenTypes.USE_VAR
        if name[0].isupper(): return TokenTypes.USE_TYPE

        return TokenTypes.ATTR


    def tokenize_qualifier(self, node, base_type):

        access = node.split(".")

        lastaccess = len(access) -1
        for i, qual in enumerate(access):
            if i == 0:
                self.state.add(qual, base_type(qual) if callable(base_type) else base_type)
            else:
                self.state.add(qual, TokenTypes.ATTR)
            if i != lastaccess: self.state.add(".")

    # Elements --------------------------------

    def tokenize_Annotation(self, node):
        pass

    def tokenize_ElementValuePair(self, node):
        pass

    def tokenize_ElementArrayValue(self, node):
        pass

    # Types ------------------------------------

    def tokenize_BasicType(self, node):
        self.state.add(node.name, TokenTypes.USE_TYPE)

        for d in node.dimensions:
            if d is None:
                self.state.add("[]")
            else:
                self.state.add("[")
                self.tokenize(d)
                self.state.add("]")

    def tokenize_ReferenceType(self, node):
        self.state.add(node.name, TokenTypes.USE_TYPE)
        
        if node.arguments is not None:
            with self.brackets("<", ">"):
                self.tokenize_List(node.arguments)


        for d in node.dimensions:
            if d is None:
                self.state.add("[]")
            else:
                with self.brackets("[", "]"):
                    self.tokenize(d)


    def tokenize_TypeParameter(self, node):
        pass


    # Parameter --------------------------------

    def tokenize_FormalParameter(self, node):
        
        for annotation in node.annotations:
            self.tokenize(annotation)
        
        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)
        
        self.tokenize(node.type)
        self.state.add(node.name, TokenTypes.DEF_VAR)
        self.scope.add(node.name)

        if node.varargs: self.state.add("*")

    def tokenize_InferredFormalParameter(self, node):
        pass

    def tokenize_LocalVariableDeclaration(self, node):
        pass

    def tokenize_VariableDeclarator(self, node):
        pass


    # Statements --------------------------------

    def tokenize_IfStatement(self, node):
        pass

    def tokenize_WhileStatement(self, node):
        pass

    def tokenize_DoStatement(self, node):
        pass

    def tokenize_ForStatement(self, node):
        pass

    def tokenize_AssertStatement(self, node):
        pass

    def tokenize_BreakStatement(self, node):
        pass

    def tokenize_ContinueStatement(self, node):
        pass

    def tokenize_ReturnStatement(self, node):
        pass

    def tokenize_ThrowStatement(self, node):
        pass

    def tokenize_SynchronizedStatement(self, node):
        pass

    def tokenize_TryStatement(self, node):
        pass

    def tokenize_SwitchStatement(self, node):
        pass

    def tokenize_BlockStatement(self, node):
        pass

    def tokenize_StatementExpression(self, node):

        if node.label is not None:
            self.state.add(node.label, TokenTypes.NAME)
            self.state.add(":")

        self.tokenize(node.expression)
        self.state.add(";")

    # Try-Catch ----------------------------------

    def tokenize_TryResource(self, node):
        pass

    def tokenize_CatchClause(self, node):
        pass

    def tokenize_CatchClauseParameter(self, node):
        pass

    # Control ------------------------------------

    def tokenize_SwitchStatementCase(self, node):
        pass

    def tokenize_ForControl(self, node):
        pass

    def tokenize_EnhancedForControl(self, node):
        pass

    # Expression ---------------------------------

    def tokenize_MethodInvocation(self, node):

        with self.pre_post(node):

            if node.qualifier is not None:
                self.tokenize_qualifier(node.qualifier, self._identify_name) 
                self.state.add(".")

            self.state.add(node.member, TokenTypes.CALL_FUNC)

            with self.brackets():
                self.tokenize_List(node.arguments)

        

    def tokenize_SuperMethodInvocation(self, node):
        with self.pre_post(node):

            self.state.add("super", TokenTypes.KEYWORDS)
            self.state.add(".")

            if node.qualifier is not None:
                self.tokenize_qualifier(node.qualifier, self._identify_name)
                self.state.add(".")
                
            self.state.add(node.member, TokenTypes.CALL_FUNC)

            with self.brackets():
                self.tokenize_List(node.arguments)


    def tokenize_Assignment(self, node):
        
        with self.state.new_ctx("DEF"):
            self.tokenize(node.expressionl)

        self.state.add(node.type)

        with self.state.new_ctx("USE"):
            self.tokenize(node.value)


    def tokenize_TernaryExpression(self, node):
        pass

    def tokenize_BinaryOperation(self, node):
        pass

    def tokenize_Cast(self, node):
        pass

    def tokenize_MethodReference(self, node):
        pass

    def tokenize_LambdaExpression(self, node):
        pass

    # Basics ------------------------------------
    
    def tokenize_Literal(self, node):
        
        with self.pre_post(node):
            
            if "\"" in node.value or "\'" in node.value:
                lit_id = TokenTypes.LIT_STRING
            else:
                lit_id = TokenTypes.LIT_NUMBER
            
            self.state.add(node.value, lit_id)


    def tokenize_This(self, node):
        with self.pre_post(node):

            self.state.add("this", TokenTypes.KEYWORDS)

            if node.selectors is not None:
                for selector in node.selectors:
                    self.state.add(".")
                    self.tokenize(selector)

    def tokenize_MemberReference(self, node):
        
        with self.pre_post(node):
            member_name = node.member
            if node.qualifier is not None and len(node.qualifier) > 0: 
                member_name = node.qualifier + "." + member_name
            self.tokenize_qualifier(member_name, self._identify_name)


    def tokenize_SuperMemberReference(self, node):
        with self.pre_post(node):

            self.state.add("super", TokenTypes.KEYWORDS)
            self.state.add(".")
            self.state.add(node.member, TokenTypes.ATTR)

            if node.selectors is not None:
                for selector in node.selectors:
                    self.state.add(".")
                    self.tokenize(selector)

    def tokenize_ArraySelector(self, node):
        pass

    def tokenize_ClassReference(self, node):
        pass

    def tokenize_VoidClassReference(self, node):
        pass

    # Creater -----------------------------------

    def tokenize_ArrayCreator(self, node):
        pass

    def tokenize_InnerClassCreator(self, node):
        pass

    # Declaration --------------------------------

    def tokenize_MethodDeclaration(self, node):

        for annotation in node.annotations:
            self.tokenize(annotation)
        
        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)

        if node.return_type is None:
            self.state.add("void", TokenTypes.USE_TYPE)
        else:
            self.state.add(node.return_type, TokenTypes.USE_TYPE)

        self.state.add(node.name, TokenTypes.DEF_FUNC)

        with self.brackets():
            self.tokenize_List(node.parameters)

        if node.throws is not None: 

            self.state.add("throws", TokenTypes.KEYWORDS)

            lastthrow = len(node.throws) - 1
            for i, throw in enumerate(node.throws):
                self.state.add(throw, TokenTypes.USE_TYPE)
                if i != lastthrow: self.state.add(",")
        
        with self.brackets("{", "}"):
            for stmt in node.body:
                self.tokenize(stmt)



def method_tokenize(code):
    
    tree = javalang.parse.parse(code)

    method_defs = {}

    for _, node in tree.filter(javalang.tree.MethodDeclaration):

        state = TokenizationState()
        NodeTokenizer(state)(node)

        method_defs[node.name] = {
            "tokens": state.tokens,
            "types" : state.type_ids
        }
    
    return method_defs
