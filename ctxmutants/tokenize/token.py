from enum import IntEnum


class TokenTypes(IntEnum):

    SYNTAX    = 0
    KEYWORDS  = 1
    DEF_FUNC  = 2
    CALL_FUNC = 3
    DEF_VAR   = 4
    USE_VAR   = 5
    UOP       = 6
    BOP       = 7
    USE_TYPE  = 8
    ATTR      = 9
    LIT_STRING= 10
    LIT_NUMBER= 11
    NAME      = 12
    ASSIGN    = 13




class _TokenContext:

    def __init__(self, state, ctx_obj):
        self.state = state
        self.ctx_obj = ctx_obj

    def __enter__(self):
        self.state.ctx_stack.append(self.ctx_obj)
    
    def __exit__(self, type, value, traceback):
        assert self.state.ctx_stack[-1] == self.ctx_obj
        self.state.ctx_stack.pop()


class TokenizationState:

    def __init__(self):
        self.tokens   = []
        self.type_ids = []

        self.ctx_stack = []

    def add(self, token, type_id = TokenTypes.SYNTAX):
        self.tokens.append(token)
        self.type_ids.append(type_id)

    def indent(self): self.add("#INDENT#")

    def dedent(self): self.add("#UNINDENT#")

    def newline(self): self.add("#NEWLINE#")

    def new_ctx(self, ctx_obj):
        return _TokenContext(self, ctx_obj)

    @property
    def context(self):
        return None if len(self.ctx_stack) == 0 else self.ctx_stack[-1]
        
