#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def write_scp(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for k,p in rows:
            f.write(f'{k} {p}\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples-jsonl', type=Path, required=True)
    ap.add_argument('--out-bundle', type=Path, required=True)
    args=ap.parse_args()
    args.out_bundle.mkdir(parents=True, exist_ok=True)
    scp=args.out_bundle/'scp'; scp.mkdir(exist_ok=True)
    mix=[]; ref=[]; aux=[]; inter=[]; meta=[]
    with open(args.samples_jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            key=r['key']
            target_spk, interferer_spk = [str(x) for x in r['spk']]
            target_ref=r['src'][target_spk][0]
            interferer_ref=r['src'][interferer_spk][0]
            # Use an enrollment from cues audio resource would be ideal; for this stress bundle,
            # choose the target source itself? No: avoid leakage. Use source_meta target_source's sibling cannot be guaranteed.
            # Existing USEF bundle used external enrollment. Here reuse manifest cue if available is not in sample, so derive first non-target utterance from v2 cues audio.json later would be overkill.
            # For now use target_ref as aux would leak and inflate performance; instead we read cues/audio.json unavailable here? handled below.
            meta.append((r, key, target_spk, interferer_spk, target_ref, interferer_ref))
    # load audio resource if present next to samples path
    audio_json=args.samples_jsonl.parent/'cues'/'audio.json'
    audio_res=json.load(open(audio_json, encoding='utf-8')) if audio_json.exists() else {}
    metadata=[]
    for r,key,target_spk,interferer_spk,target_ref,interferer_ref in meta:
        enroll_items=audio_res.get(f'{key}::{target_spk}', [])
        if not enroll_items:
            # hard fail would be cleaner, but keep explicit marker
            enrollment=target_ref
            leakage_warning=True
        else:
            enrollment=enroll_items[0]['path']
            leakage_warning=False
        mix_path=r['mix']['default'][0]
        mix.append((key,mix_path)); ref.append((key,target_ref)); aux.append((key,enrollment)); inter.append((key,interferer_ref))
        rr={
            'key':key, 'condition':r.get('condition','similar_speaker_hard_v2'), 'mix':mix_path,
            'target_ref':target_ref, 'interferer_ref':interferer_ref, 'enrollment':enrollment,
            'target_spk':target_spk, 'interferer_spk':interferer_spk,
            'speaker_similarity':r.get('speaker_similarity'), 'speaker_similarity_rank':r.get('speaker_similarity_rank'),
            'speaker_similarity_detail':r.get('speaker_similarity_detail'), 'same_gender':r.get('same_gender'),
            'pair_id':r.get('pair_id'), 'source_meta':r.get('source_meta',{}), 'enrollment_leakage_warning':leakage_warning
        }
        metadata.append(rr)
    write_scp(scp/'mix.scp', mix); write_scp(scp/'ref.scp', ref); write_scp(scp/'aux.scp', aux); write_scp(scp/'interferer.scp', inter)
    with open(args.out_bundle/'metadata.jsonl','w',encoding='utf-8') as f:
        for r in metadata: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('bundle', args.out_bundle, 'n', len(metadata), 'leakage_warnings', sum(r['enrollment_leakage_warning'] for r in metadata))
if __name__ == '__main__': main()
