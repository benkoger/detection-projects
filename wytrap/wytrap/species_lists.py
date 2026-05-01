"""Wyoming wildlife species lists for zero-shot BioCLIP classification.

Lists are common names because pybioclip's CustomLabelsClassifier uses them
verbatim as text prompts. Researchers can fork these or pass a path to a
newline-delimited file via the CLI's --species flag.
"""

# Wyoming mammals — the Wyoming Game & Fish list, common-name form.
WYOMING_MAMMALS: list[str] = [
    "American badger",
    "American beaver",
    "American bison",
    "American marten",
    "American mink",
    "American pika",
    "big brown bat",
    "bighorn sheep",
    "black bear",
    "black-footed ferret",
    "black-tailed prairie dog",
    "bobcat",
    "bushy-tailed woodrat",
    "Canada lynx",
    "Columbian ground squirrel",
    "common muskrat",
    "cougar",
    "coyote",
    "deer mouse",
    "domestic cat",
    "domestic dog",
    "elk",
    "ermine",
    "fisher",
    "fox squirrel",
    "golden-mantled ground squirrel",
    "gray fox",
    "gray wolf",
    "grizzly bear",
    "hoary bat",
    "house mouse",
    "least chipmunk",
    "least weasel",
    "little brown bat",
    "long-tailed weasel",
    "meadow vole",
    "moose",
    "mountain cottontail",
    "mountain goat",
    "mule deer",
    "muskrat",
    "Norway rat",
    "northern flying squirrel",
    "northern pocket gopher",
    "northern raccoon",
    "Nuttall's cottontail",
    "porcupine",
    "pronghorn",
    "raccoon",
    "red fox",
    "red squirrel",
    "river otter",
    "rock squirrel",
    "snowshoe hare",
    "striped skunk",
    "swift fox",
    "thirteen-lined ground squirrel",
    "Uinta ground squirrel",
    "western jumping mouse",
    "white-footed mouse",
    "white-tailed deer",
    "white-tailed jackrabbit",
    "white-tailed prairie dog",
    "wolverine",
    "wyoming ground squirrel",
    "yellow-bellied marmot",
]

# Wyoming reptiles + amphibians (combined; small list for the state).
WYOMING_REPTILES_AMPHIBIANS: list[str] = [
    "boreal chorus frog",
    "boreal toad",
    "bullsnake",
    "common garter snake",
    "eastern racer",
    "great basin spadefoot",
    "great plains toad",
    "greater short-horned lizard",
    "milk snake",
    "northern leopard frog",
    "ornate box turtle",
    "painted turtle",
    "plains hognose snake",
    "plains spadefoot",
    "prairie rattlesnake",
    "rubber boa",
    "sagebrush lizard",
    "smooth greensnake",
    "snapping turtle",
    "spiny softshell",
    "tiger salamander",
    "valley garter snake",
    "wandering garter snake",
    "western terrestrial garter snake",
    "wood frog",
]

# Wyoming birds — abridged starter list (~60 of the most commonly trapped
# species). The full Wyoming bird checklist is ~440 species; expand this as
# needed. Keep camera-trap-realistic birds (ground-dwelling, raptors,
# corvids, large waterfowl) over rare fly-overs.
WYOMING_BIRDS: list[str] = [
    "American crow",
    "American kestrel",
    "American magpie",
    "American robin",
    "American white pelican",
    "American wigeon",
    "bald eagle",
    "barn owl",
    "black-billed magpie",
    "black-capped chickadee",
    "blue grouse",
    "Brewer's blackbird",
    "burrowing owl",
    "California gull",
    "Canada goose",
    "chukar",
    "Clark's nutcracker",
    "common loon",
    "common merganser",
    "common raven",
    "Cooper's hawk",
    "downy woodpecker",
    "dusky grouse",
    "ferruginous hawk",
    "golden eagle",
    "great blue heron",
    "great horned owl",
    "greater sage-grouse",
    "gray jay",
    "gray partridge",
    "Hungarian partridge",
    "killdeer",
    "long-billed curlew",
    "mallard",
    "merlin",
    "mountain bluebird",
    "mountain chickadee",
    "mourning dove",
    "northern flicker",
    "northern goshawk",
    "northern harrier",
    "northern shoveler",
    "osprey",
    "peregrine falcon",
    "pied-billed grebe",
    "pinyon jay",
    "prairie falcon",
    "red-tailed hawk",
    "ring-necked pheasant",
    "rock pigeon",
    "rough-legged hawk",
    "ruffed grouse",
    "sage thrasher",
    "sandhill crane",
    "sharp-shinned hawk",
    "sharp-tailed grouse",
    "snowy owl",
    "Steller's jay",
    "Swainson's hawk",
    "trumpeter swan",
    "turkey vulture",
    "western meadowlark",
    "white-tailed ptarmigan",
    "wild turkey",
    "wood duck",
]

WYOMING_ALL: list[str] = (
    WYOMING_MAMMALS + WYOMING_BIRDS + WYOMING_REPTILES_AMPHIBIANS
)


# Testbed list for the YNP-BisonGraze validation harness. The species here
# match the post-merge categories in helpers/helpers.py so we can compare
# BioCLIP outputs to the trained Faster R-CNN baseline directly.
YNP_TESTBED: list[str] = [
    "American badger",
    "bighorn sheep",
    "American bison",
    "black bear",
    "grizzly bear",
    "coyote",
    "gray wolf",
    "red fox",
    "elk",
    "human",
    "moose",
    "mule deer",
    "white-tailed deer",
    "pronghorn",
    "bird",
    "rodent",
]


BUILTIN_LISTS: dict[str, list[str]] = {
    "wyoming_all": WYOMING_ALL,
    "wyoming_mammals": WYOMING_MAMMALS,
    "wyoming_birds": WYOMING_BIRDS,
    "wyoming_reptiles_amphibians": WYOMING_REPTILES_AMPHIBIANS,
    "ynp_testbed": YNP_TESTBED,
}


def load_species(spec: str) -> list[str]:
    """Resolve a --species argument to a concrete list.

    `spec` is either a builtin name (e.g. 'wyoming_all') or a path to a
    newline-delimited text file. Lines starting with '#' and blank lines
    are ignored.
    """
    if spec in BUILTIN_LISTS:
        return list(BUILTIN_LISTS[spec])

    with open(spec, "r") as f:
        names = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        raise ValueError(f"Species file {spec!r} contained no species.")
    return names
