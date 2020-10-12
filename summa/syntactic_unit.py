class SyntacticUnit(object):

    def __init__(self, text, token=None, tag=None, source=None):
        self.text = text
        self.token = token
        self.tag = tag[:2] if tag else None  # just first two letters of tag
        self.index = -1
        self.score = -1
        self.source = source

    def __str__(self):
        return "Original unit: '" + self.text + "' *-*-*-* " + "Processed unit: '" + self.token + "'"

    def __repr__(self):
        return str(self)
