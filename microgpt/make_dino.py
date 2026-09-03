"""
Generate input_dino.txt: real dinosaur genus names (lowercased) for pretraining.

Names sourced from Wikipedia's "List of dinosaur genera"
(https://en.wikipedia.org/wiki/List_of_dinosaur_genera). Curated to valid,
well-known genera with strong regular suffixes (-saurus, -raptor, -ceratops,
-titan, -mimus, etc.) so a tiny 16-dim model can latch onto the pattern.

Pretrain context is block_size=16 and each doc is tokenized as [BOS] + chars,
so names must be <= 15 characters to fit without truncation.

Run:  python3 make_dino.py
"""

import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Real dinosaur genera (from Wikipedia list), <= 15 chars, clean ASCII lowercase.
names = [
    # -saurus
    "allosaurus", "stegosaurus", "spinosaurus", "megalosaurus", "apatosaurus",
    "ceratosaurus", "kentrosaurus", "nodosaurus", "ankylosaurus", "plateosaurus",
    "camarasaurus", "titanosaurus", "seismosaurus", "supersaurus", "brontosaurus",
    "barosaurus", "cetiosaurus", "dryosaurus", "ouranosaurus", "edmontosaurus",
    "hadrosaurus", "gryposaurus", "kritosaurus", "styracosaurus", "torosaurus",
    "sinosaurus", "abelisaurus", "aucasaurus", "noasaurus", "mapusaurus",
    "rajasaurus", "majungasaurus", "masiakasaurus", "dromaeosaurus", "adasaurus",
    "byronosaurus", "tarbosaurus", "eotyrannus", "riojasaurus", "anchisaurus",
    "mussaurus", "sarcosaurus", "elasmosaurus", "plesiosaurus", "mosasaurus",
    "tylosaurus", "kronosaurus", "shonisaurus", "camptosaurus", "thescelosaurus",
    "wuerhosaurus", "sauropelta", "edmontonia", "panoplosaurus", "gastonia",
    "euoplocephalus", "talarurus", "pinacosaurus", "saichania", "scelidosaurus",
    "scutellosaurus", "emausaurus", "lesothosaurus", "coelophysis", "megaraptor",
    "neovenator", "baryonyx", "suchomimus", "irritator", "concavenator",
    "carnotaurus", "rugops", "shunosaurus", "datousaurus", "nigersaurus",
    "europasaurus", "giraffatitan", "vulcanodon", "gasosaurus", "monolophosaurus",
    "cryolophosaurus", "dilophosaurus", "guanlong", "dilong", "yutyrannus",
    "gorgosaurus", "albertosaurus", "alioramus", "lythronax", "herrerasaurus",
    "eoraptor", "staurikosaurus", "tawa", "massospondylus", "lufengosaurus",
    "melanorosaurus", "yunnanosaurus", "isanosaurus", "jobaria", "amargasaurus",
    "diplodocus", "brachiosaurus", "triceratops",
    # -raptor
    "velociraptor", "utahraptor", "microraptor", "buitreraptor", "bambiraptor",
    "atrociraptor", "austroraptor", "gigantoraptor", "oviraptor", "pyroraptor",
    "dakotaraptor", "acheroraptor", "linheraptor", "variraptor", "vayuraptor",
    "fukuiraptor", "condorraptor", "megaraptor", "sinraptor", "overoraptor",
    # -ceratops / ceratopsians
    "protoceratops", "pentaceratops", "chasmosaurus", "centrosaurus", "zuniceratops",
    "psittacosaurus", "einiosaurus", "achelousaurus", "diabloceratops", "kosmoceratops",
    "vagaceratops", "avaceratops", "medusaceratops", "nasutoceratops", "wendiceratops",
    "regaliceratops", "sinoceratops", "spiclypeus", "yamaceratops", "liaoceratops",
    # -mimus (ornithomimosaurs)
    "gallimimus", "struthiomimus", "ornithomimus", "garudimimus", "harpymimus",
    "pelecanimimus", "sinornithomimus", "qiupalong", "deinocheirus", "shenzhousaurus",
    # -titan (titanosaurs)
    "paralititan", "dreadnoughtus", "argentinosaurus", "puertasaurus", "rapetosaurus",
    "saltasaurus", "isisaurus", "malawisaurus", "bonitasaura", "patagotitan",
    "notocolossus", "mansourasaurus", "nemegtosaurus", "opisthocoelic",
    # pachycephalosaurs / others
    "stegoceras", "homalocephale", "prenocephale", "dracorex", "stygimoloch",
    "goyocephale", "tylocephale", "wannanosaurus", "yinlong", "chaoyangsaurus",
    # therizinosaurs / oviraptors / troodontids
    "nothronychus", "beipiaosaurus", "alxasaurus", "erlikosaurus", "segnosaurus",
    "falcarius", "citipati", "conchoraptor", "khaan", "anzu",
    "troodon", "zanabazar", "mei", "byronosaurus", "sinovenator",
    "sinornithoides", "mononykus", "shuvuuia", "patagonykus", "albinykus",
    # feathered / early birds-adjacent
    "compsognathus", "sinosauropteryx", "juravenator", "scipionyx", "caudipteryx",
    "protarchaeopteryx", "incisivosaurus", "avimimus", "epidexipteryx", "yi",
    "anchiornis", "microraptor", "sinornithosaurus", "changyuraptor", "zhenyuanlong",
    # pterosaurs & marine (commonly grouped with dinos in kids' sets)
    "pteranodon", "pterodactylus", "dimorphodon", "rhamphorhynchus", "ichthyosaurus",
    "ophthalmosaurus", "temnodontosaurus", "quetzalcoatlus", "tapejara", "dsungaripterus",
]

# Deduplicate, keep clean ASCII lowercase, filter to <= 15 chars.
MAX_LEN = 15
clean = []
too_long = []
seen = set()
for n in names:
    n = n.lower().strip()
    if n and n.isascii() and n.isalpha() and n not in seen:
        seen.add(n)
        if len(n) <= MAX_LEN:
            clean.append(n)
        else:
            too_long.append(n)

with open(os.path.join(DATA_DIR, "input_dino.txt"), "w") as f:
    for n in clean:
        f.write(n + "\n")

lengths = [len(n) for n in clean]
print(f"Wrote {len(clean)} dinosaur names to {DATA_DIR}/input_dino.txt (<= {MAX_LEN} chars)")
print(f"Length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
if too_long:
    print(f"Filtered out {len(too_long)} names > {MAX_LEN} chars:")
    for n in sorted(too_long, key=len, reverse=True):
        print(f"  {len(n):2d}  {n}")
