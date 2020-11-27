import os


ref = os.environ['REF']
hypo = os.environ['HYPO']
src = os.environ['SRC']

assert os.path.exists(ref)
assert os.path.exists(hypo)
assert os.path.exists(src)

ref_o = f"{ref}.filtered"
hypo_o = f"{hypo}.filtered"
src_o = f"{src}.filtered"
matched_idx_o = f"{src}.matched_idx"

orefs = []
ohypos = []
osrcs = []

duplicates = 0
dup_idx = []
with open(ref, "r") as rf:
    with open(hypo, "r") as hf:
        with open(src, "r") as sf:
            refs = rf.read().strip().split("\n")
            hypos = hf.read().strip().split("\n")
            srcs = sf.read().strip().split("\n")

            for i, (r, h, s) in enumerate(zip(refs, hypos, srcs)):
                if r.lower() == h.lower():
                    if i % 1000 == 0:
                        print(f'dup: {i}')
                    duplicates += 1
                    dup_idx.append(i)
                else:
                    orefs.append(r)
                    ohypos.append(h)
                    osrcs.append(s)

print(f"duplicates: {duplicates}")
# print(f"dup_idx: {dup_idx}")

with open(matched_idx_o, 'w') as f:
    f.write('\n'.join([str(x) for x in dup_idx]))

print(f"save: {matched_idx_o}")
with open(ref_o, "w") as f:
    f.write("\n".join(orefs))
print(f"save: {ref_o}")
with open(hypo_o, "w") as f:
    f.write("\n".join(ohypos))
print(f"save: {hypo_o}")
with open(src_o, "w") as f:
    f.write("\n".join(osrcs))
print(f"save: {src_o}")



