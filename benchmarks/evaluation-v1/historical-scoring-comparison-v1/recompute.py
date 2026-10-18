"""Recompute archived relevance, latency and process RSS without running a database."""
from pathlib import Path, PurePosixPath
from fractions import Fraction
import argparse, collections, gzip, hashlib, json, math, re

def require(value, reason):
    if not value: raise ValueError(reason)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def quantiles(values):
    require(values and all(type(v) in [int,float] and math.isfinite(v) and v>=0 for v in values),'invalid latency samples')
    ordered=sorted(values)
    return {'samples':len(ordered),'p50_ms':ordered[(len(ordered)+1)//2-1],'p95_ms':ordered[math.ceil(0.95*len(ordered))-1]}

def equal_numeric(actual, expected, path='root'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'comparison keys '+path)
        for key in expected:equal_numeric(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'comparison length '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):equal_numeric(a,b,path+f'[{i}]')
    elif type(expected) in [int,float]:
        require(type(actual) in [int,float] and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10),'comparison value '+path)
    else:require(actual==expected,'comparison value '+path)

def recompute(directory, qrels_path, output):
    directory=directory.resolve(strict=True)
    require(not output.exists() and not output.is_symlink(),'output already exists')
    index=json.loads((directory/'trial-index.json').read_text())
    qrels={};positives=collections.defaultdict(set)
    with qrels_path.open() as stream:
        for line in stream:
            row=json.loads(line);key=(row['query_id'],row['chunk_external_id'])
            require(key not in qrels and type(row['relevance']) is int and row['relevance']>=0,'invalid/duplicate qrel')
            qrels[key]=row['relevance']
            if row['relevance']>=1:positives[key[0]].add(key[1])
    require(len(qrels)==3960 and len(positives)==48,'fixed quality judgment coverage')
    require(digest(qrels_path)==index['quality_qrels_sha256'],'qrels digest changed')
    files=json.loads((directory/'lineage.json').read_text())['files'];known={}
    for row in files:
        relative=PurePosixPath(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==row['path'] and row['path'] not in known,'unsafe/duplicate manifest path')
        path=directory/row['path']
        require(all(not part.is_symlink() for part in [path,*path.parents] if part!=directory.parent) and path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'changed file: '+row['path'])
        if row.get('storage')=='gzip':
            count=0;expanded=hashlib.sha256()
            with gzip.open(path,'rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    count+=len(block);require(count<=row['uncompressed_bytes'],'expanded file exceeds bound');expanded.update(block)
            require(count==row['uncompressed_bytes'] and expanded.hexdigest()==row['uncompressed_sha256'],'compressed payload changed')
        known[row['path']]=row
    require('trial-index.json' in known and 'comparisons.json' in known,'missing comparison/index identity')
    trials=index['trials'];require(len(trials)==36 and len({t['trial_id'] for t in trials})==36,'trial coverage')
    profiles=list(dict.fromkeys(t['profile'] for t in trials));require(len(profiles)==3,'budget setting coverage')
    expected_trials=[]
    for scope in ['quality','scale']:
        for diagnostic in [False,True]:
            for round_ in ([1] if diagnostic else [1,2]):
                for profile in profiles:
                    for policy in (['window_scores','complete_U'] if round_==1 else ['complete_U','window_scores']):
                        expected_trials.append({'scope':scope,'diagnostic':diagnostic,'round':round_,'profile':profile,'policy':policy})
    require(all({k:t[k] for k in e}==e and t['trial_id']==f'trial-{i:02}' for i,(t,e) in enumerate(zip(trials,expected_trials))),'trial plan/order mismatch')
    quality=collections.defaultdict(list);scale=collections.defaultdict(list);rss=collections.defaultdict(list);counts=collections.Counter()
    for trial in trials:
        stdout=directory/trial['stdout'];stderr=directory/trial['stderr']
        require(trial['stdout'] in known and trial['stderr'] in known,'unbound observation stream')
        peak=re.findall(r'(\d+)\s+maximum resident set size',stderr.read_text());require(len(peak)==1,'RSS record missing/ambiguous');peak=int(peak[0])
        start={};results={};terminal={};group_keys=set();expected_samples={}
        if trial['scope']=='quality':
            for mode in ['text','vector','hybrid']:
                for query in positives:
                    for cache in ['fresh_handle','warm_same_handle']:expected_samples[f'quality:{mode}:{query}:{cache}']={'query_id':query,'mode':mode,'cache':cache}
        else:
            queries={'text':'scale-text-764','vector':'scale-vector-e0','hybrid':'scale-hybrid-764-e0','hybrid-filtered':'scale-hybrid-filtered-764-e0-project4'}
            for mode,query in queries.items():
                for api in ['rust_search_execute','rust_search_standard_json']:
                    for phase,n in [('first_query',1),('warmup',0 if trial['diagnostic'] else 2),('measured',1 if trial['diagnostic'] else 20)]:
                        for iteration in range(1,n+1):expected_samples[f'scale:{mode}:{api}:{phase}:{iteration}']={'query_id':query,'mode':mode,'scope':api,'phase':phase,'round':iteration}
        with stdout.open() as stream:
            for line in stream:
                row=json.loads(line)
                event=row.get('event')
                if event not in ['sample_start','sample_result','sample_terminal']:continue
                key=(row['sample_id'],row['attempt']);require(row['trial_id']==trial['trial_id'] and type(row['attempt']) is int and row['attempt']==1 and key[0] in expected_samples and row['query_id']==expected_samples[key[0]]['query_id'],'sample identity')
                if event=='sample_start':require(key not in start and key not in results and key not in terminal and type(row['monotonic_ns']) is int,'duplicate/invalid start');start[key]=row['monotonic_ns'];continue
                if event=='sample_terminal':
                    require(key in results and key not in terminal and row['status']=='success' and row['api_elapsed_scope']=='completed_api_scope','terminal identity/status')
                    terminal[key]=(row['monotonic_ns'],row['api_elapsed_ms']);continue
                require(key in start and key not in terminal and key not in results,'duplicate/out-of-order result')
                require(all(row.get(k)==v for k,v in expected_samples[key[0]].items()),'result scope mismatch')
                require(row['profile']==trial['profile'].replace('-','_') and row.get('policy',trial['policy'])==trial['policy'],'profile/policy identity')
                elapsed=row['elapsed_ms'];require(type(elapsed) in [float,int] and math.isfinite(elapsed) and elapsed>=0,'invalid elapsed time')
                results[key]=elapsed
                if trial['scope']=='quality':
                    query=row['query_id'];hits=row['result']['results'];ids=[h['chunk_external_id'] for h in hits]
                    require(query in positives and len(ids)<=10 and len(ids)==len(set(ids)) and all((query,k) in qrels for k in ids),'invalid/unjudged returned IDs')
                    recall=Fraction(len(set(ids)&positives[query]),len(positives[query]))
                    mrr=next((Fraction(1,i) for i,k in enumerate(ids,1) if k in positives[query]),Fraction())
                    if not trial['diagnostic']:
                        group=(trial['profile'],row['mode'],row['cache'],trial['policy']);quality[group].append((recall,mrr,elapsed));group_keys.add(group)
                elif trial['scope']=='scale':
                    if not trial['diagnostic'] and row['phase']=='measured':
                        group=(trial['profile'],row['mode'],row['scope'],trial['policy']);scale[group].append(elapsed);group_keys.add(group)
                else:raise ValueError('unknown trial scope')
        require(len(results)==trial['expected_records']==len(expected_samples) and set(start)==set(results)==set(terminal)=={(k,1) for k in expected_samples},'missing/extra sample events')
        for key,elapsed in results.items():
            require(start[key]<=terminal[key][0] and terminal[key][1]==elapsed,'event time mismatch')
        counts[(trial['scope'],'diagnostic' if trial['diagnostic'] else 'primary')]+=len(results)
        for group in group_keys:rss[(trial['scope'],*group)].append(peak)
    expected_counts={('quality','primary'):3456,('quality','diagnostic'):1728,('scale','primary'):2208,('scale','diagnostic'):96}
    require(dict(counts)==expected_counts,'planned sample coverage')
    original=json.loads((directory/'comparisons.json').read_text());computed={'scope':original['scope'],'quality':[],'scale':[]}
    require(len(original['quality'])==18 and len(original['scale'])==24,'comparison matrix coverage')
    require(len({(r['profile'],r['mode'],r['cache']) for r in original['quality']})==18 and len({(r['profile'],r['mode'],r['scope']) for r in original['scale']})==24,'duplicate comparison cell')
    require(len(quality)==36 and len(scale)==48,'complete grouped observations')
    for kind,groups in [('quality',quality),('scale',scale)]:
        for record in original[kind]:
            axis=record['cache'] if kind=='quality' else record['scope'];pair={}
            for role in ['window_scores','complete_U']:
                key=(record['profile'],record['mode'],axis,role);values=groups[key]
                require(len(values)==(96 if kind=='quality' else 40),'comparison group sample count')
                if kind=='quality':
                    pair[role]={'recall10':float(sum((v[0] for v in values),Fraction())/len(values)), 'mrr10':float(sum((v[1] for v in values),Fraction())/len(values)), 'latency':quantiles([v[2] for v in values]),'rss_each_process':rss[(kind,*key)]}
                else:pair[role]={'latency':quantiles(values),'rss_each_process':rss[(kind,*key)]}
            result={'profile':record['profile'],'mode':record['mode'],'cache' if kind=='quality' else 'scope':axis,'versions':pair}
            if kind=='quality':result.update(recall_delta=pair['complete_U']['recall10']-pair['window_scores']['recall10'],mrr_delta=pair['complete_U']['mrr10']-pair['window_scores']['mrr10'])
            else:result['p95_delta_ms']=pair['complete_U']['latency']['p95_ms']-pair['window_scores']['latency']['p95_ms']
            computed[kind].append(result)
    equal_numeric(computed,original)
    result={'schema':1,'status':'ARCHIVED_METRICS_RECOMPUTED','scope':'Hash/event coverage, macro relevance, pooled latency quantiles and recorded process RSS only; not engine score correctness, semantic certification, physical IO or resource-budget proof.','source_report_sha256':index['source_report_sha256'],'qrels_sha256':digest(qrels_path),'sample_counts':{a+'_'+b:n for (a,b),n in counts.items()},'comparison':computed}
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'query_samples':sum(counts.values()),'quality_groups':18,'scale_groups':24}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observations-dir',type=Path,required=True);parser.add_argument('--qrels',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();recompute(args.observations_dir,args.qrels,args.output)
