class Person():
    """
    This module implements an agent-based model.
    It defines a class which represents a person, with an internal state which is
    one of S, I or R.

    By default, a person is susceptible.  They become infected using the infect method,
    and recovered by recover method.

    """

    def __init__(self):
        self.S = True # default setting not I not R -> person is susceptible
        self.E = False
        self.I = False  # if I = True, the person has infected
        self.R = False  # if R = True, the person has recovered and cannot be infected

    def is_infected(self):
        """
        returns true if the person has infected
        """
        return self.I

    def is_exposed(self):
        return self.E

    def is_recovered(self):
        """
        returns true if the person has recovered
        """
        return self.R

    def contact(self):
        """
        among all contacts, susceptible person get exposed
        """
        if self.S:
            self.exposed()

    def exposed(self):
        self.S = False
        self.E = True

    def infect(self):
        """
        exposed individuals become infectious
        """
        if not self.E:
            raise ValueError("only exposed person can befome infectious")
        self.I = True
        self.E = False

    def recover(self):
        """
        infectious persons get recovered and cannot get infected later
        """
        if not self.I:
            raise ValueError("only infectious persons can recover")
        self.R = True
        self.I = False
