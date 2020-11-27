import os
import argparse


ref = os.environ['REF']
# hypo = os.environ['HYPO']
# src = os.environ['SRC']

assert os.path.exists(ref)
# assert os.path.exists(hypo)
# assert os.path.exists(src)

oref = os.environ['REF_TW']
assert oref is not None and oref != ""

count = 0
outs = []

with open(ref, "r") as rf:
    sents = rf.read().strip().split("\n")

    for s in sents:
        if "„" in s or '"' in s:
            s = s.replace("„", "&quot;").replace('"', "&quot;")
            count += 1
        outs.append(s)

print(f"replace count: {count}")

with open(oref, "w") as f:
    f.write('\n'.join(outs))