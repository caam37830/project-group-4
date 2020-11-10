



class Person():
    """
    This module implements an agent-based model.
    It defines a class which represents a person, with an internal state which is
    one of S, I or R.

    By default, a person is not susceptible.  They become infected using the infect method,
    and recovered by recover method.

    """


    def __init__(self):
        # self.S = True # default setting not I not R -> person is susceptible
        self.I = False  # if I = True, the person has infected
        self.R = False  # if R = True, the person has recovered and cannot be infected

    def is_infected(self):
        """
        returns true if the person has infected
        """
        return self.I

    def is_recovered(self):
        """
        returns true if the person has recovered
        """
        return self.R

    def infect(self):
        """
        person get infected
        """
        self.I = True

    def recover(self):
        """
        person get recovered and cannot get infected later
        """
        self.R = True
        self.I = False