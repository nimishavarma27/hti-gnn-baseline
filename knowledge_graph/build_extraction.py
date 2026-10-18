"""Validate the 8 HTI tables against each other and write a graphify extraction
built directly from the rows (no LLM, every edge comes from a table row)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from graphify.ids import make_id

D = Path(__file__).resolve().parent / 'data'
OUT = Path(__file__).resolve().parent / 'graphify-out' / '.graphify_extract.json'
CHECKS = Path(__file__).resolve().parent / 'data_checks.txt'

log = []
problems = []


def say(msg):
    print(msg)
    log.append(msg)


def check(cond, msg):
    say(('OK    ' if cond else 'FAIL  ') + msg)
    if not cond:
        problems.append(msg)


# ---------------- load ----------------
herbs = pd.read_csv(D / 'herbs.tsv', sep='\t', dtype=str)
cmp_meta = pd.read_csv(D / 'compounds.tsv', sep='\t', dtype=str)
ecfp = pd.read_csv(D / 'compound_features_ecfp4.csv')
aac = pd.read_csv(D / 'target_features_aac.csv')
hc = pd.read_csv(D / 'herb_compound_edges.tsv', sep='\t', dtype=str)
ct = pd.read_csv(D / 'compound_target_labeled.csv', dtype=str)
hct = pd.read_csv(D / 'herb_compound_target.csv', dtype=str,
                  usecols=['herb_id', 'compound_id', 'protein_id'])
un = pd.read_csv(D / 'compound_target_unlabeled.csv', dtype=str, usecols=['compound_id', 'protein_id'])

say('== row counts')
for name, df in [('herbs.tsv', herbs), ('compounds.tsv', cmp_meta), ('compound_features_ecfp4.csv', ecfp),
                 ('target_features_aac.csv', aac), ('herb_compound_edges.tsv', hc),
                 ('compound_target_labeled.csv', ct), ('herb_compound_target.csv', hct),
                 ('compound_target_unlabeled.csv', un)]:
    say(f'      {name:32s} {len(df):>9,} rows')

# ---------------- entity universes ----------------
say('== entities')
H = set(herbs['herb_id'])
check(herbs['herb_id'].is_unique, f'herbs.tsv herb_id unique ({len(H)} herbs)')

C = set(ecfp['compound_id'])
check(ecfp['compound_id'].is_unique, f'ecfp compound_id unique ({len(C)} compounds)')
bits = ecfp.drop(columns='compound_id').to_numpy()
check(bits.shape[1] == 2048, f'ecfp has 2048 bit columns ({bits.shape[1]})')
check(np.isin(bits, [0, 1]).all(), 'ecfp values are all 0/1')

check(cmp_meta['compound_id'].is_unique, f'compounds.tsv compound_id unique ({cmp_meta["compound_id"].nunique()})')
check(set(cmp_meta['compound_id']) == C, 'compounds.tsv covers exactly the ecfp compound universe')

T = set(aac['protein_id'])
check(aac['protein_id'].is_unique, f'aac protein_id unique ({len(T)} targets)')
check((aac['protein_id'] == 'UNIPROT:' + aac['uniprot_accession']).all(), 'protein_id = UNIPROT:<accession> for every target')
aac_sum = aac.filter(like='aac_').sum(axis=1)
check(np.allclose(aac_sum, 1.0, atol=1e-6), f'aac rows sum to 1 (min {aac_sum.min():.6f}, max {aac_sum.max():.6f})')

# ---------------- herb-compound ----------------
say('== herb_compound_edges.tsv')
check(set(hc['herb_id']) <= H, 'every herb_id is in herbs.tsv')
check(set(hc['compound_id']) <= C, 'every compound_id is in the compound universe')
check((hc['compound_id'] == hc['IMPPAT_Phytochemical_identifier']).all(), 'compound_id matches IMPPAT_Phytochemical_identifier')
check((hc['herb_id'] == 'IMPPAT_PLANT:' + hc['Plant_identifier']).all(), 'herb_id matches IMPPAT_PLANT:<Plant_identifier>')
hc['part'] = hc['plant_part'].fillna('').str.strip().str.lower()
hc_pairs = hc.groupby(['herb_id', 'compound_id'])['part'].apply(lambda s: sorted({p for p in s if p}))
hc_pairs = hc_pairs.reset_index()
say(f'      {len(hc)} rows -> {len(hc_pairs)} unique herb-compound pairs, '
    f'{hc_pairs["herb_id"].nunique()} herbs, {hc_pairs["compound_id"].nunique()} compounds')
check(hc_pairs['compound_id'].nunique() == len(C), 'every compound belongs to at least one herb')
check(hc_pairs['herb_id'].nunique() == len(H), 'every herb has at least one compound')

# ---------------- compound-target ----------------
say('== compound_target_labeled.csv')
check(not ct.duplicated(['compound_id', 'protein_id']).any(), 'no duplicate compound-target pairs')
check((ct['label'] == '1').all(), 'every label is 1 (positives only)')
check(set(ct['compound_id']) <= C, 'every compound_id is in the compound universe')
check(set(ct['protein_id']) <= T, 'every protein_id is in the target universe')
check(set(ct['protein_id']) == T, 'every target has at least one known compound')
say(f'      {len(ct)} pairs, {ct["compound_id"].nunique()} compounds with a known target, '
    f'{len(C) - ct["compound_id"].nunique()} compounds with none')

# ---------------- unlabeled ----------------
say('== compound_target_unlabeled.csv')
pos_key = set(ct['compound_id'] + '|' + ct['protein_id'])
un_key = un['compound_id'] + '|' + un['protein_id']
check(un_key.is_unique, 'no duplicate unlabeled pairs')
check(set(un['compound_id']) <= C and set(un['protein_id']) <= T, 'all unlabeled ids are in the universes')
check(not un_key.isin(pos_key).any(), 'unlabeled and labeled pairs do not overlap')
check(len(un) + len(ct) == len(C) * len(T),
      f'labeled + unlabeled = every compound x target pair ({len(ct)} + {len(un)} = {len(C)} x {len(T)} = {len(C) * len(T)})')
del un, un_key

# ---------------- herb-compound-target ----------------
say('== herb_compound_target.csv')
hc_key = set(hc_pairs['herb_id'] + '|' + hc_pairs['compound_id'])
check((hct['herb_id'] + '|' + hct['compound_id']).isin(hc_key).all(), 'every (herb, compound) is a real herb-compound edge')
check((hct['compound_id'] + '|' + hct['protein_id']).isin(pos_key).all(), 'every (compound, target) is a labeled positive')
derived = hc_pairs.merge(ct[['compound_id', 'protein_id']], on='compound_id')
ht_from_join = set(derived['herb_id'] + '|' + derived['protein_id'])
ht_file = set(hct['herb_id'] + '|' + hct['protein_id'])
check(ht_file == ht_from_join, f'herb-target pairs equal the H-C x C-T join ({len(ht_file)} vs {len(ht_from_join)})')
say(f'      {hct["herb_id"].nunique()} herbs and {hct["compound_id"].nunique()} compounds appear; '
    f'{len(ht_file)} unique herb-target pairs (derived, not added as edges)')

if problems:
    say(f'\n{len(problems)} check(s) failed, graph NOT written')
    CHECKS.write_text('\n'.join(log), encoding='utf-8')
    sys.exit(1)

# ---------------- build extraction ----------------
def src(name):
    return str(D / name)

F_HERB, F_CMP, F_TGT = src('herbs.tsv'), src('compounds.tsv'), src('target_features_aac.csv')
F_HC, F_CT = src('herb_compound_edges.tsv'), src('compound_target_labeled.csv')


def nid(*parts):
    return make_id(*parts)


def unique_labels(names, ids):
    """Keep the name, add the id only where two entities share the same name."""
    s = pd.Series(names, index=ids).fillna('')
    s = s.where(s.str.strip() != '', pd.Series(ids, index=ids))
    dup = s.duplicated(keep=False)
    return {i: (f'{n} ({i})' if d else n) for i, n, d in zip(s.index, s.values, dup.values)}


nodes, edges = [], []

h_lab = unique_labels(herbs['botanical_name'].values, herbs['herb_id'].values)
for _, r in herbs.iterrows():
    nodes.append({
        'id': nid('herbs', r['herb_id'].split(':')[-1]), 'label': h_lab[r['herb_id']],
        'file_type': 'concept', 'source_file': F_HERB, 'source_location': None,
        'entity_type': 'herb', 'entity_id': r['herb_id'],
        'family': r['Family'], 'common_name': r['Common_name'] if isinstance(r['Common_name'], str) else None,
        'system_of_medicine': r['System_of_Medicine'] if isinstance(r['System_of_Medicine'], str) else None,
    })

fams = sorted(herbs['Family'].dropna().unique())
for f in fams:
    nodes.append({'id': nid('herbs', 'family', f), 'label': f'Family: {f}', 'file_type': 'concept',
                  'source_file': F_HERB, 'source_location': None, 'entity_type': 'plant_family'})
for _, r in herbs.dropna(subset=['Family']).iterrows():
    edges.append({'source': nid('herbs', r['herb_id'].split(':')[-1]), 'target': nid('herbs', 'family', r['Family']),
                  'relation': 'belongs_to_family', 'confidence': 'EXTRACTED', 'confidence_score': 1.0,
                  'source_file': F_HERB, 'source_location': None, 'weight': 1.0})

cm = cmp_meta.set_index('compound_id')
known_cmp = set(ct['compound_id'])
c_lab = unique_labels(cm['Phytochemical name_standardised'].reindex(sorted(C)).values, np.array(sorted(C)))
n_bits = dict(zip(ecfp['compound_id'], bits.sum(axis=1).astype(int)))
for c in sorted(C):
    nodes.append({
        'id': nid('compounds', c), 'label': c_lab[c], 'file_type': 'concept', 'source_file': F_CMP,
        'source_location': None, 'entity_type': 'compound', 'entity_id': c,
        'pubchem_cid': cm.at[c, 'Pubchem_CID'] if isinstance(cm.at[c, 'Pubchem_CID'], str) else None,
        'ecfp4_bits_on': int(n_bits[c]), 'has_known_target': c in known_cmp,
    })

sym = ct.drop_duplicates('protein_id').set_index('protein_id')['hgnc_symbol']
t_names = [f'{sym.get(p)} ({n})' if isinstance(sym.get(p), str) else n
           for p, n in zip(aac['protein_id'], aac['protein_name'])]
t_lab = unique_labels(t_names, aac['protein_id'].values)
for _, r in aac.iterrows():
    p = r['protein_id']
    nodes.append({
        'id': nid('target_features_aac', r['uniprot_accession']), 'label': t_lab[p], 'file_type': 'concept',
        'source_file': F_TGT, 'source_location': None, 'entity_type': 'target', 'entity_id': p,
        'protein_name': r['protein_name'], 'hgnc_symbol': sym.get(p) if isinstance(sym.get(p), str) else None,
    })

for _, r in hc_pairs.iterrows():
    edges.append({'source': nid('herbs', r['herb_id'].split(':')[-1]), 'target': nid('compounds', r['compound_id']),
                  'relation': 'contains', 'confidence': 'EXTRACTED', 'confidence_score': 1.0,
                  'source_file': F_HC, 'source_location': None, 'weight': 1.0,
                  'plant_parts': r['part'] or None})

acc = dict(zip(aac['protein_id'], aac['uniprot_accession']))
for _, r in ct.iterrows():
    edges.append({'source': nid('compounds', r['compound_id']), 'target': nid('target_features_aac', acc[r['protein_id']]),
                  'relation': 'binds_target', 'confidence': 'EXTRACTED', 'confidence_score': 1.0,
                  'source_file': F_CT, 'source_location': None, 'weight': 1.0,
                  'evidence_source': r['evidence_source']})

ids = [n['id'] for n in nodes]
check(len(ids) == len(set(ids)), f'all {len(ids)} node ids unique')
idset = set(ids)
check(all(e['source'] in idset and e['target'] in idset for e in edges), f'all {len(edges)} edges point to existing nodes')
labels = [n['label'] for n in nodes]
check(len(labels) == len(set(labels)), 'all node labels unique')
keys = [(e['source'], e['target']) for e in edges]
check(len(keys) == len(set(keys)), 'no duplicate edges')

counts = pd.Series([n['entity_type'] for n in nodes]).value_counts().to_dict()
say(f'== extraction: {len(nodes)} nodes {counts}, {len(edges)} edges '
    f'(contains {len(hc_pairs)}, binds_target {len(ct)}, belongs_to_family {len(edges) - len(hc_pairs) - len(ct)})')

if problems:
    say(f'\n{len(problems)} check(s) failed, graph NOT written')
    CHECKS.write_text('\n'.join(log), encoding='utf-8')
    sys.exit(1)

OUT.write_text(json.dumps({'nodes': nodes, 'edges': edges, 'hyperedges': [], 'input_tokens': 0, 'output_tokens': 0},
                          ensure_ascii=False), encoding='utf-8')
CHECKS.write_text('\n'.join(log), encoding='utf-8')
say(f'wrote {OUT}')
