# from Cortex import Cortex

# cortex = Cortex()

# response = cortex.process_query("A mechanic charges different rates to repair the tires of trucks and cars.  For each truck tire that is repaired, the mechanic will charge $60 and for each car tire that is repaired, the mechanic will charge $40.  On Thursday, the mechanic repairs 6 truck tires and 4 car tires.  On Friday, the mechanic repairs 12 car tries and doesn't repair any truck tires.  How much more revenue did the mechanic earn on the day with higher revenue?")
# print(response)

from CortexSingle import CortexSingle

cortex = CortexSingle()

response = cortex.process_query("Generate a coherent sentence using all these concepts: animal, pit, tourist, arch, shoe, tube, chair, card, lay, store, knit, sheep, trim, kid, camel, cardboard, remove, part, game, cart, hockey, rink, sail, vest, beverage, examine, market, tool, branch, burn.")
print(response)