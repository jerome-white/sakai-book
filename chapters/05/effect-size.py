import sys
import csv
import math

from irstats import VarianceSystems

systems = VarianceSystems(sys.stdin)
deviation = math.sqrt(systems.V())

fieldnames = [
    'system_1',
    'system_2',
    'effect',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
for (i, j) in systems.differences():
    #
    # Equation 5.17
    #
    row = (*i, j / deviation)
    writer.writerow(dict(zip(fieldnames, row)))
