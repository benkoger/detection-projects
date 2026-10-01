"""Wyoming camera-trap detection + classification pipeline.

MegaDetector (class-agnostic animal localizer) -> one of several species
classifiers (BioCLIP 2 zero-shot, SpeciesNet, AddaxAI zoo models), all
restricted to the same taxon-node vocabulary and writing the same records.
"""

__version__ = "0.2.0"
