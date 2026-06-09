#!/usr/bin/env python3
import json, os
from pathlib import Path
BASE=Path('/data/tse_cue_project/diagnostics/bsrnn_ft13_hard_simspk_v3_20260605')
CONDS=['sir0db','sirm3db','sirm5db']
CUES=['usef','tfmap','context']

def ensure_link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src,dst)

for cond in CONDS:
    rows=[]
    for part in ['part0','part1']:
        split=BASE/'splits'/f'{cond}_{part}.jsonl'
        for line in split.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            r=json.loads(line); key=r['key']; target=str(r['spk'][0]); inter=str(r['spk'][1])
            rows.append({
                'key':key,
                'target_spk':target,
                'interferer_spk':inter,
                'target_ref':r['src'][target][0],
                'interferer_ref':r['src'][inter][0],
                'mix_path':r['mix']['default'][0],
                'part':part,
            })
    meta=BASE/f'bsrnn_hard_simspk_v3_{cond}_speaker_diag_metadata.jsonl'
    with meta.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('wrote metadata', meta, len(rows))
    by_key={r['key']:r for r in rows}
    for cue in CUES:
        made=0
        for part in ['part0','part1']:
            audio_dir=BASE/f'{cue}_{cond}_{part}'/'audio'
            link_dir=BASE/'est_by_key'/f'{cue}_{cond}'
            for row in [r for r in rows if r['part']==part]:
                key=row['key']; target=row['target_spk']
                matches=list(audio_dir.glob(f'*-{key}-T{target}.wav'))
                if len(matches)!=1:
                    matches=list(audio_dir.glob(f'*{key}*T{target}.wav'))
                if len(matches)!=1:
                    raise RuntimeError(f'{cue} {cond} {part} {key} T{target}: got {len(matches)} matches')
                ensure_link(str(matches[0]), link_dir/f'{key}.wav')
                made+=1
        print('links', cue, cond, made)

