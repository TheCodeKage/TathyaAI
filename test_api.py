from enum import Enum

class Domain(Enum):
    MEDICAL = 0
    LEGAL = 1
    FINANCE = 2
    ACADEMIC = 3
    GENERAL = 4

    @staticmethod
    def from_str(s: str):
        return Domain[s.upper()]

domain = Domain.from_str(input("Enter domain: "))

if domain == Domain.MEDICAL:
