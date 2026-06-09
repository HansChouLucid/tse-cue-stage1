#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from statistics import mean, median

ROOTS = {
    'wav2vec2': ('content_ssl', 'ssl_content_per_chunk_1s.csv'),
    'wavlm': ('content_wavlm', 'wavlm_base_plus_per_chunk_1s.csv'),
    'hubert': ('content_hubert', 'hubert_base_per_chunk_1s.csv'),
}

def read_rows(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def fnum(x, default=0.0):
    try: return float(x)
    except Exception: return default

def inum(x):
    try: return int(float(x))
    except Exception: return 0

def stats(vals):
    vals=list(vals)
    if not vals:
        return {'n':0,'mean':None,'median':None,'min':None,'max':None}
    vals_sorted=sorted(vals)
    def pct(p):
        if not vals_sorted: return None
        k=(len(vals_sorted)-1)*p
        lo=int(k); hi=min(lo+1,len(vals_sorted)-1); frac=k-lo
        return vals_sorted[lo]*(1-frac)+vals_sorted[hi]*frac
    return {'n':len(vals),'mean':mean(vals),'median':median(vals),'p10':pct(.10),'p25':pct(.25),'p75':pct(.75),'p90':pct(.90),'min':vals_sorted[0],'max':vals_sorted[-1]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--cue', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args=ap.parse_args()
    by_verifier={}
    for verifier,(root,fn) in ROOTS.items():
        p=args.base/root/args.cue/fn
        if not p.exists():
            raise FileNotFoundError(f'missing {verifier}: {p}')
        rows=read_rows(p)
        d={}
        for r in rows:
            k=(r['key'], int(r['chunk_idx']))
            d[k]=r
        by_verifier[verifier]=d
        print(args.cue, verifier, len(d), p)
    common=set.intersection(*(set(d) for d in by_verifier.values()))
    if not common:
        raise RuntimeError('no common chunks')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_path=args.out_dir/'content_multiverifier_per_chunk_1s.csv'
    fields=['key','chunk_idx','target_spk','interferer_spk','content_similarity','start_sec','end_sec',
            'wav2vec2_content_gap','wavlm_content_gap','hubert_content_gap',
            'wav2vec2_content_mismatch','wavlm_content_mismatch','hubert_content_mismatch',
            'local_target_interferer_si_sdr_gap','waveform_prefers_interferer',
            'two_verifier_content_mismatch','two_verifier_si_sdr_confirmed_drift',
            'tri_verifier_content_mismatch','tri_verifier_si_sdr_confirmed_drift']
    utt={}
    with chunk_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for k in sorted(common):
            base=by_verifier['wav2vec2'][k]
            mm={name: inum(by_verifier[name][k].get('content_mismatch',0)) for name in ROOTS}
            gap={name: fnum(by_verifier[name][k].get('content_gap',0)) for name in ROOTS}
            si_gap=fnum(base.get('local_si_sdr_gap',0))
            wave=1 if si_gap < 0 else inum(base.get('waveform_prefers_interferer',0))
            two=1 if (mm['wav2vec2'] and mm['wavlm']) else 0
            tri=1 if (mm['wav2vec2'] and mm['wavlm'] and mm['hubert']) else 0
            two_drift=1 if two and si_gap < 0 else 0
            tri_drift=1 if tri and si_gap < 0 else 0
            out={
                'key':k[0], 'chunk_idx':k[1], 'target_spk':base.get('target_spk',''), 'interferer_spk':base.get('interferer_spk',''),
                'content_similarity':base.get('content_similarity','normal_basic'), 'start_sec':base.get('start_sec',''), 'end_sec':base.get('end_sec',''),
                'wav2vec2_content_gap':gap['wav2vec2'], 'wavlm_content_gap':gap['wavlm'], 'hubert_content_gap':gap['hubert'],
                'wav2vec2_content_mismatch':mm['wav2vec2'], 'wavlm_content_mismatch':mm['wavlm'], 'hubert_content_mismatch':mm['hubert'],
                'local_target_interferer_si_sdr_gap':si_gap, 'waveform_prefers_interferer':wave,
                'two_verifier_content_mismatch':two, 'two_verifier_si_sdr_confirmed_drift':two_drift,
                'tri_verifier_content_mismatch':tri, 'tri_verifier_si_sdr_confirmed_drift':tri_drift,
            }
            w.writerow(out)
            u=utt.setdefault(k[0], {'key':k[0], 'target_spk':base.get('target_spk',''), 'interferer_spk':base.get('interferer_spk',''),
                                    'chunks':0, 'sis':[], 'w2v':0, 'wavlm':0, 'hubert':0, 'two':0, 'two_drift':0, 'tri':0, 'tri_drift':0, 'wave':0})
            u['chunks']+=1; u['sis'].append(si_gap); u['w2v']+=mm['wav2vec2']; u['wavlm']+=mm['wavlm']; u['hubert']+=mm['hubert']
            u['two']+=two; u['two_drift']+=two_drift; u['tri']+=tri; u['tri_drift']+=tri_drift; u['wave']+=wave
    utt_path=args.out_dir/'content_multiverifier_per_utterance_1s.csv'
    ufields=['key','target_spk','interferer_spk','num_chunks','mean_local_target_interferer_si_sdr_gap','waveform_interferer_rate',
             'wav2vec2_content_mismatch_rate','wavlm_content_mismatch_rate','hubert_content_mismatch_rate',
             'two_verifier_content_mismatch_rate','two_verifier_si_sdr_confirmed_drift_rate',
             'tri_verifier_content_mismatch_rate','tri_verifier_si_sdr_confirmed_drift_rate',
             'any_two_verifier_content_mismatch','any_two_verifier_si_sdr_confirmed_drift','any_tri_verifier_content_mismatch','any_tri_verifier_si_sdr_confirmed_drift',
             'two_verifier_content_mismatch_ge20pct','two_verifier_si_sdr_confirmed_drift_ge20pct','tri_verifier_content_mismatch_ge20pct','tri_verifier_si_sdr_confirmed_drift_ge20pct']
    with utt_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=ufields); w.writeheader()
        for u in sorted(utt.values(), key=lambda x:x['key']):
            n=u['chunks']
            row={'key':u['key'], 'target_spk':u['target_spk'], 'interferer_spk':u['interferer_spk'], 'num_chunks':n,
                 'mean_local_target_interferer_si_sdr_gap':mean(u['sis']) if u['sis'] else 0,
                 'waveform_interferer_rate':u['wave']/n,
                 'wav2vec2_content_mismatch_rate':u['w2v']/n, 'wavlm_content_mismatch_rate':u['wavlm']/n, 'hubert_content_mismatch_rate':u['hubert']/n,
                 'two_verifier_content_mismatch_rate':u['two']/n, 'two_verifier_si_sdr_confirmed_drift_rate':u['two_drift']/n,
                 'tri_verifier_content_mismatch_rate':u['tri']/n, 'tri_verifier_si_sdr_confirmed_drift_rate':u['tri_drift']/n,
                 'any_two_verifier_content_mismatch':1 if u['two']>0 else 0, 'any_two_verifier_si_sdr_confirmed_drift':1 if u['two_drift']>0 else 0,
                 'any_tri_verifier_content_mismatch':1 if u['tri']>0 else 0, 'any_tri_verifier_si_sdr_confirmed_drift':1 if u['tri_drift']>0 else 0,
                 'two_verifier_content_mismatch_ge20pct':1 if u['two']/n>=0.2 else 0, 'two_verifier_si_sdr_confirmed_drift_ge20pct':1 if u['two_drift']/n>=0.2 else 0,
                 'tri_verifier_content_mismatch_ge20pct':1 if u['tri']/n>=0.2 else 0, 'tri_verifier_si_sdr_confirmed_drift_ge20pct':1 if u['tri_drift']/n>=0.2 else 0}
            w.writerow(row)
    rows=read_rows(utt_path)
    summary={
        'cue':args.cue,
        'num_utterances':len(rows),
        'num_chunks':len(common),
        'chunk_rates':{},
        'utterance_rate_distributions':{},
        'case_level_rates':{},
    }
    for col in ['wav2vec2_content_mismatch','wavlm_content_mismatch','hubert_content_mismatch','waveform_prefers_interferer','two_verifier_content_mismatch','two_verifier_si_sdr_confirmed_drift','tri_verifier_content_mismatch','tri_verifier_si_sdr_confirmed_drift']:
        vals=[]
        with chunk_path.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f): vals.append(fnum(r[col]))
        summary['chunk_rates'][col]=mean(vals)
    for col in ['waveform_interferer_rate','wav2vec2_content_mismatch_rate','wavlm_content_mismatch_rate','hubert_content_mismatch_rate','two_verifier_content_mismatch_rate','two_verifier_si_sdr_confirmed_drift_rate','tri_verifier_content_mismatch_rate','tri_verifier_si_sdr_confirmed_drift_rate','mean_local_target_interferer_si_sdr_gap']:
        summary['utterance_rate_distributions'][col]=stats(fnum(r[col]) for r in rows)
    for col in ['any_two_verifier_content_mismatch','any_two_verifier_si_sdr_confirmed_drift','any_tri_verifier_content_mismatch','any_tri_verifier_si_sdr_confirmed_drift','two_verifier_content_mismatch_ge20pct','two_verifier_si_sdr_confirmed_drift_ge20pct','tri_verifier_content_mismatch_ge20pct','tri_verifier_si_sdr_confirmed_drift_ge20pct']:
        summary['case_level_rates'][col]=mean(fnum(r[col]) for r in rows)
    (args.out_dir/'content_multiverifier_summary_1s.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
