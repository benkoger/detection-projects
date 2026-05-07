"""Wyoming wildlife species lists for zero-shot BioCLIP classification.

Each species is a dict ``{"scientific": "...", "common": "..."}``. BioCLIP-2
was trained on TreeOfLife-200M, where labels are taxonomic, so passing
binomials as text prompts performs noticeably better than common names.
We keep both forms: the scientific name is what BioCLIP sees, the common
name is what humans (and ground-truth labels) see.

Researchers can fork these lists or pass a path to a text file via
``--species``. The text-file format is one species per line, either:

    Canis latrans | coyote
    coyote                    # used as both scientific and common

Lines starting with ``#`` and blank lines are ignored.
"""

from __future__ import annotations

from typing import TypedDict


class Species(TypedDict):
    scientific: str
    common: str


def _sp(scientific: str, common: str | None = None) -> Species:
    return {"scientific": scientific, "common": common or scientific}


# ----------------------------- Mammals -----------------------------
WYOMING_MAMMALS: list[Species] = [
    _sp("Taxidea taxus", "American badger"),
    _sp("Castor canadensis", "American beaver"),
    _sp("Bison bison", "American bison"),
    _sp("Martes americana", "American marten"),
    _sp("Neogale vison", "American mink"),
    _sp("Ochotona princeps", "American pika"),
    _sp("Eptesicus fuscus", "big brown bat"),
    _sp("Ovis canadensis", "bighorn sheep"),
    _sp("Ursus americanus", "black bear"),
    _sp("Mustela nigripes", "black-footed ferret"),
    _sp("Cynomys ludovicianus", "black-tailed prairie dog"),
    _sp("Lynx rufus", "bobcat"),
    _sp("Neotoma cinerea", "bushy-tailed woodrat"),
    _sp("Lynx canadensis", "Canada lynx"),
    _sp("Urocitellus columbianus", "Columbian ground squirrel"),
    _sp("Puma concolor", "cougar"),
    _sp("Canis latrans", "coyote"),
    _sp("Peromyscus maniculatus", "deer mouse"),
    _sp("Felis catus", "domestic cat"),
    _sp("Canis familiaris", "domestic dog"),
    _sp("Cervus canadensis", "elk"),
    _sp("Mustela erminea", "ermine"),
    _sp("Pekania pennanti", "fisher"),
    _sp("Sciurus niger", "fox squirrel"),
    _sp("Callospermophilus lateralis", "golden-mantled ground squirrel"),
    _sp("Urocyon cinereoargenteus", "gray fox"),
    _sp("Canis lupus", "gray wolf"),
    _sp("Ursus arctos", "grizzly bear"),
    _sp("Lasiurus cinereus", "hoary bat"),
    _sp("Mus musculus", "house mouse"),
    _sp("Neotamias minimus", "least chipmunk"),
    _sp("Mustela nivalis", "least weasel"),
    _sp("Myotis lucifugus", "little brown bat"),
    _sp("Neogale frenata", "long-tailed weasel"),
    _sp("Microtus pennsylvanicus", "meadow vole"),
    _sp("Alces alces", "moose"),
    _sp("Oreamnos americanus", "mountain goat"),
    _sp("Odocoileus hemionus", "mule deer"),
    _sp("Ondatra zibethicus", "muskrat"),
    _sp("Rattus norvegicus", "Norway rat"),
    _sp("Glaucomys sabrinus", "northern flying squirrel"),
    _sp("Thomomys talpoides", "northern pocket gopher"),
    _sp("Sylvilagus nuttallii", "Nuttall's cottontail"),
    _sp("Erethizon dorsatum", "porcupine"),
    _sp("Antilocapra americana", "pronghorn"),
    _sp("Procyon lotor", "raccoon"),
    _sp("Vulpes vulpes", "red fox"),
    _sp("Tamiasciurus hudsonicus", "red squirrel"),
    _sp("Lontra canadensis", "river otter"),
    _sp("Otospermophilus variegatus", "rock squirrel"),
    _sp("Lepus americanus", "snowshoe hare"),
    _sp("Mephitis mephitis", "striped skunk"),
    _sp("Vulpes velox", "swift fox"),
    _sp("Ictidomys tridecemlineatus", "thirteen-lined ground squirrel"),
    _sp("Urocitellus armatus", "Uinta ground squirrel"),
    _sp("Zapus princeps", "western jumping mouse"),
    _sp("Peromyscus leucopus", "white-footed mouse"),
    _sp("Odocoileus virginianus", "white-tailed deer"),
    _sp("Lepus townsendii", "white-tailed jackrabbit"),
    _sp("Cynomys leucurus", "white-tailed prairie dog"),
    _sp("Gulo gulo", "wolverine"),
    _sp("Urocitellus elegans", "Wyoming ground squirrel"),
    _sp("Marmota flaviventris", "yellow-bellied marmot"),
]

# ----------------------------- Birds -----------------------------
WYOMING_BIRDS: list[Species] = [
    _sp("Corvus brachyrhynchos", "American crow"),
    _sp("Falco sparverius", "American kestrel"),
    _sp("Turdus migratorius", "American robin"),
    _sp("Pelecanus erythrorhynchos", "American white pelican"),
    _sp("Mareca americana", "American wigeon"),
    _sp("Haliaeetus leucocephalus", "bald eagle"),
    _sp("Tyto alba", "barn owl"),
    _sp("Pica hudsonia", "black-billed magpie"),
    _sp("Poecile atricapillus", "black-capped chickadee"),
    _sp("Euphagus cyanocephalus", "Brewer's blackbird"),
    _sp("Athene cunicularia", "burrowing owl"),
    _sp("Larus californicus", "California gull"),
    _sp("Branta canadensis", "Canada goose"),
    _sp("Alectoris chukar", "chukar"),
    _sp("Nucifraga columbiana", "Clark's nutcracker"),
    _sp("Gavia immer", "common loon"),
    _sp("Mergus merganser", "common merganser"),
    _sp("Corvus corax", "common raven"),
    _sp("Astur cooperii", "Cooper's hawk"),
    _sp("Dryobates pubescens", "downy woodpecker"),
    _sp("Dendragapus obscurus", "dusky grouse"),
    _sp("Buteo regalis", "ferruginous hawk"),
    _sp("Aquila chrysaetos", "golden eagle"),
    _sp("Ardea herodias", "great blue heron"),
    _sp("Bubo virginianus", "great horned owl"),
    _sp("Centrocercus urophasianus", "greater sage-grouse"),
    _sp("Perisoreus canadensis", "Canada jay"),
    _sp("Perdix perdix", "gray partridge"),
    _sp("Charadrius vociferus", "killdeer"),
    _sp("Numenius americanus", "long-billed curlew"),
    _sp("Anas platyrhynchos", "mallard"),
    _sp("Falco columbarius", "merlin"),
    _sp("Sialia currucoides", "mountain bluebird"),
    _sp("Poecile gambeli", "mountain chickadee"),
    _sp("Zenaida macroura", "mourning dove"),
    _sp("Colaptes auratus", "northern flicker"),
    _sp("Astur atricapillus", "northern goshawk"),
    _sp("Circus hudsonius", "northern harrier"),
    _sp("Spatula clypeata", "northern shoveler"),
    _sp("Pandion haliaetus", "osprey"),
    _sp("Falco peregrinus", "peregrine falcon"),
    _sp("Podilymbus podiceps", "pied-billed grebe"),
    _sp("Gymnorhinus cyanocephalus", "pinyon jay"),
    _sp("Falco mexicanus", "prairie falcon"),
    _sp("Buteo jamaicensis", "red-tailed hawk"),
    _sp("Phasianus colchicus", "ring-necked pheasant"),
    _sp("Columba livia", "rock pigeon"),
    _sp("Buteo lagopus", "rough-legged hawk"),
    _sp("Bonasa umbellus", "ruffed grouse"),
    _sp("Oreoscoptes montanus", "sage thrasher"),
    _sp("Antigone canadensis", "sandhill crane"),
    _sp("Accipiter striatus", "sharp-shinned hawk"),
    _sp("Tympanuchus phasianellus", "sharp-tailed grouse"),
    _sp("Bubo scandiacus", "snowy owl"),
    _sp("Cyanocitta stelleri", "Steller's jay"),
    _sp("Buteo swainsoni", "Swainson's hawk"),
    _sp("Cygnus buccinator", "trumpeter swan"),
    _sp("Cathartes aura", "turkey vulture"),
    _sp("Sturnella neglecta", "western meadowlark"),
    _sp("Lagopus leucura", "white-tailed ptarmigan"),
    _sp("Meleagris gallopavo", "wild turkey"),
    _sp("Aix sponsa", "wood duck"),
]

# ----------------------------- Reptiles + amphibians -----------------------------
WYOMING_REPTILES_AMPHIBIANS: list[Species] = [
    _sp("Pseudacris maculata", "boreal chorus frog"),
    _sp("Anaxyrus boreas", "boreal toad"),
    _sp("Pituophis catenifer", "bullsnake"),
    _sp("Thamnophis sirtalis", "common garter snake"),
    _sp("Coluber constrictor", "eastern racer"),
    _sp("Spea intermontana", "great basin spadefoot"),
    _sp("Anaxyrus cognatus", "great plains toad"),
    _sp("Phrynosoma hernandesi", "greater short-horned lizard"),
    _sp("Lampropeltis triangulum", "milk snake"),
    _sp("Lithobates pipiens", "northern leopard frog"),
    _sp("Terrapene ornata", "ornate box turtle"),
    _sp("Chrysemys picta", "painted turtle"),
    _sp("Heterodon nasicus", "plains hognose snake"),
    _sp("Spea bombifrons", "plains spadefoot"),
    _sp("Crotalus viridis", "prairie rattlesnake"),
    _sp("Charina bottae", "rubber boa"),
    _sp("Sceloporus graciosus", "sagebrush lizard"),
    _sp("Opheodrys vernalis", "smooth greensnake"),
    _sp("Chelydra serpentina", "snapping turtle"),
    _sp("Apalone spinifera", "spiny softshell"),
    _sp("Ambystoma mavortium", "tiger salamander"),
    _sp("Thamnophis elegans", "western terrestrial garter snake"),
    _sp("Lithobates sylvaticus", "wood frog"),
]

WYOMING_ALL: list[Species] = (
    WYOMING_MAMMALS + WYOMING_BIRDS + WYOMING_REPTILES_AMPHIBIANS
)


# --------------------------- YNP_TESTBED (default) ---------------------------
#
# Hybrid family-level testbed: species-level prompts for animals we want to
# identify precisely (the named YNP ungulates, badger, human), and family- or
# genus-level prompts for the merged catch-all categories (Canid / Bear /
# Deer / rodent / bird). Drastically reduces score-splitting: the rodent
# vote isn't fragmented across 11 species, the bird vote isn't fragmented
# across 18.
#
# BioCLIP-2 was trained on TreeOfLife-200M, which annotates at every
# taxonomic rank (kingdom → ... → species), so prompts like "Canidae" and
# "Sciuridae" get meaningful activations.
#
# When you DO want species-level identification (e.g. distinguishing wolf
# from coyote from fox), use YNP_TESTBED_FULL (below) which has all 48
# species expanded.
YNP_TESTBED: list[Species] = [
    # Singleton ungulates — species-level keeps them separable from each
    # other and from the merged classes.
    _sp("Bison bison", "bison"),
    _sp("Cervus canadensis", "elk"),
    _sp("Alces alces", "moose"),
    _sp("Antilocapra americana", "pronghorn"),
    _sp("Ovis canadensis", "bighorn sheep"),

    # Other singletons
    _sp("Taxidea taxus", "badger"),
    _sp("Homo sapiens", "human"),

    # Merged classes — family / genus level.
    # Canidae covers coyote, gray wolf, red/swift/gray fox.
    _sp("Canidae", "Canid"),
    # Ursidae covers black bear + grizzly bear.
    _sp("Ursidae", "Bear"),
    # Odocoileus (genus) covers mule deer + white-tailed deer.
    # NOT Cervidae here — that would conflict with elk + moose singletons.
    _sp("Odocoileus", "Deer"),

    # Catch-all categories — most-common Wyoming families.
    # Sciuridae: marmots, ground squirrels, tree squirrels, chipmunks
    # (most rodent sightings on YNP cameras).
    _sp("Sciuridae", "rodent"),
    # Cricetidae: voles + native mice.
    _sp("Cricetidae", "rodent"),

    # Birds — camera-trap-realistic family list. Each maps to "bird" at
    # eval time; the multiple prompts give BioCLIP separable family-level
    # embeddings without bloating to 18 species.
    _sp("Corvidae", "bird"),       # ravens, crows, magpies, jays
    _sp("Accipitridae", "bird"),   # hawks, eagles
    _sp("Phasianidae", "bird"),    # grouse, turkeys, partridges
    _sp("Anatidae", "bird"),       # waterfowl
    _sp("Strigidae", "bird"),      # owls
    _sp("Turdidae", "bird"),       # thrushes (incl. American robin)
    _sp("Gruidae", "bird"),        # cranes (incl. sandhill)
]


# --------------------------- YNP_TESTBED_FULL ---------------------------
#
# Original 48-species expansion of the merged categories. Every member
# species of Canid/Bear/Deer/rodent/bird gets its own prompt, and all
# collapse to the merged class at eval time via YNP_EVAL_MERGES.
#
# Use this when you want species-level identification (e.g. distinguishing
# wolf from coyote in the JSON output), at the cost of more score-splitting
# on supercategories.
YNP_TESTBED_FULL: list[Species] = [
    # Singleton ungulates
    _sp("Bison bison", "bison"),
    _sp("Cervus canadensis", "elk"),
    _sp("Alces alces", "moose"),
    _sp("Antilocapra americana", "pronghorn"),
    _sp("Ovis canadensis", "bighorn sheep"),

    # Canids → "Canid" at eval
    _sp("Canis latrans", "coyote"),
    _sp("Canis lupus", "gray wolf"),
    _sp("Vulpes vulpes", "red fox"),
    _sp("Vulpes velox", "swift fox"),
    _sp("Urocyon cinereoargenteus", "gray fox"),

    # Bears → "Bear" at eval
    _sp("Ursus americanus", "black bear"),
    _sp("Ursus arctos", "grizzly bear"),

    # Deer → "Deer" at eval
    _sp("Odocoileus hemionus", "mule deer"),
    _sp("Odocoileus virginianus", "white-tailed deer"),

    # Rodents (marmots, tree squirrels, ground squirrels, chipmunks,
    # voles, mice) → "rodent" at eval
    _sp("Marmota flaviventris", "yellow-bellied marmot"),
    _sp("Tamiasciurus hudsonicus", "red squirrel"),
    _sp("Sciurus niger", "fox squirrel"),
    _sp("Urocitellus elegans", "Wyoming ground squirrel"),
    _sp("Urocitellus armatus", "Uinta ground squirrel"),
    _sp("Urocitellus columbianus", "Columbian ground squirrel"),
    _sp("Callospermophilus lateralis", "golden-mantled ground squirrel"),
    _sp("Ictidomys tridecemlineatus", "thirteen-lined ground squirrel"),
    _sp("Neotamias minimus", "least chipmunk"),
    _sp("Microtus pennsylvanicus", "meadow vole"),
    _sp("Peromyscus maniculatus", "deer mouse"),

    # Birds (camera-trap-realistic: corvids, grouse, raptors, turkey)
    # → "bird" at eval
    _sp("Corvus corax", "common raven"),
    _sp("Corvus brachyrhynchos", "American crow"),
    _sp("Pica hudsonia", "black-billed magpie"),
    _sp("Perisoreus canadensis", "Canada jay"),
    _sp("Cyanocitta stelleri", "Steller's jay"),
    _sp("Nucifraga columbiana", "Clark's nutcracker"),
    _sp("Bonasa umbellus", "ruffed grouse"),
    _sp("Dendragapus obscurus", "dusky grouse"),
    _sp("Antigone canadensis", "sandhill crane"),
    _sp("Zonotrichia leucophrys", "white-crowned sparrow"),
    _sp("Meleagris gallopavo", "wild turkey"),
    _sp("Aquila chrysaetos", "golden eagle"),
    _sp("Haliaeetus leucocephalus", "bald eagle"),
    _sp("Buteo jamaicensis", "red-tailed hawk"),
    _sp("Bubo virginianus", "great horned owl"),
    _sp("Turdus migratorius", "American robin"),
    _sp("Scolopax minor", "American woodcock"),
    _sp("Haemorhous mexicanus", "house finch"),
    _sp("Poecile atricapillus", "black-capped chickadee"),
    _sp("Sturnella neglecta", "western meadowlark"),
    _sp("Sturnus vulgaris", "european starling"),

    # Singleton categories
    _sp("Taxidea taxus", "badger"),
    _sp("Homo sapiens", "human"),
]


BUILTIN_LISTS: dict[str, list[Species]] = {
    "wyoming_all": WYOMING_ALL,
    "wyoming_mammals": WYOMING_MAMMALS,
    "wyoming_birds": WYOMING_BIRDS,
    "wyoming_reptiles_amphibians": WYOMING_REPTILES_AMPHIBIANS,
    "ynp_testbed": YNP_TESTBED,            # default: family/genus + singletons
    "ynp_testbed_full": YNP_TESTBED_FULL,  # 48-species expanded version
}


def load_species(spec: str | list) -> list[Species]:
    """Resolve a --species argument to a list of Species dicts.

    `spec` is either:
      * a builtin name (e.g. 'wyoming_all'),
      * a path to a text file (one species per line, "scientific | common"
        or just one name used as both),
      * an already-resolved list of Species dicts (returned unchanged), or
      * a list of plain strings (each used as both scientific and common,
        for backward-compat).
    """
    if isinstance(spec, list):
        out: list[Species] = []
        for item in spec:
            if isinstance(item, dict):
                out.append({"scientific": item["scientific"],
                            "common": item.get("common", item["scientific"])})
            elif isinstance(item, str):
                out.append(_sp(item))
            else:
                raise TypeError(f"Unsupported species entry: {item!r}")
        return out

    if spec in BUILTIN_LISTS:
        return [dict(s) for s in BUILTIN_LISTS[spec]]

    out = []
    with open(spec, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                sci, com = (p.strip() for p in line.split("|", 1))
                out.append(_sp(sci, com))
            else:
                out.append(_sp(line))
    if not out:
        raise ValueError(f"Species file {spec!r} contained no species.")
    return out


def scientific_names(species: list[Species]) -> list[str]:
    return [s["scientific"] for s in species]


def common_names(species: list[Species]) -> list[str]:
    return [s["common"] for s in species]
