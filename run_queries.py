import os
os.environ['SWI_HOME_DIR'] = "C:\\Program Files\\swipl"

from pyswip import Prolog

prolog = Prolog()
prolog.consult("family.pl")

for result in prolog.query("grandparent(john, X)"):
    print(result)

