from pathlib import Path
import pandas as pd, json, re
SRC=Path('/data/tse_cue_project/diagnostics/real_tse_tfmap_context_20260607/bundles')
BUNDLES=[
 'tfmap_context_hsimv2_sir0db',
 'tfmap_context_hsimv2_sirm3db',
 'tfmap_context_hsimv3_sir0db',
 'tfmap_context_hsimv3_sirm3db',
 'tfmap_context_hsimv3_sirm5db',
]

def spk_from_path(p):
    # Prefer LibriSpeech-like source id if present in basename/key.
    name=Path(str(p)).stem
    m=re.search(r'__(T[12])__([0-9]+)-', name)
    if m: return m.group(2)
    # fallback: first numeric directory component before chapter in LibriSpeech paths
    parts=Path(str(p)).parts
    for i,part in enumerate(parts):
        if part.isdigit() and i+1 < len(parts) and str(parts[i+1]).isdigit():
            return part
    return ''

def interferer_spk_from_key(key):
    # Parse mixture prefix: uttA_uttB__T1/T2. T1 target=A, interferer=B; T2 target=B, interferer=A.
    prefix=key.split('__T')[0]
    m=re.search(r'__(T[12])__', key)
    if '_' in prefix and m:
        a,b=prefix.split('_',1)
        spk_a=a.split('-',1)[0]
        spk_b=b.split('-',1)[0]
        return spk_b if m.group(1)=='T1' else spk_a
    return ''

def target_spk_from_key(key):
    m=re.search(r'__T[12]__([0-9]+)-', key)
    return m.group(1) if m else ''

for b in BUNDLES:
    inf=SRC/b/'full_infer/inference_summary.csv'
    if not inf.exists():
        print('MISS',b,inf); continue
    df=pd.read_csv(inf)
    out=SRC/b/'metadata.jsonl'
    rows=[]
    with out.open('w',encoding='utf-8') as w:
        for _,r in df.iterrows():
            key=str(r['key'])
            t=target_spk_from_key(key) or spk_from_path(r['target_path'])
            itf=interferer_spk_from_key(key) or spk_from_path(r['interferer_path'])
            obj={
              'key':key,
              'target_ref':str(r['target_path']),
              'interferer_ref':str(r['interferer_path']),
              'target_spk':t,
              'interferer_spk':itf,
              'mix':str(r.get('mix_path','')),
              'est_path':str(r.get('est_path','')),
              'aux_path':str(r.get('aux_path','')),
            }
            w.write(json.dumps(obj,ensure_ascii=False)+'\n')
            rows.append(obj)
    print('WROTE',out,len(rows),'example',rows[0] if rows else None)
