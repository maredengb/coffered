"""External supervision and exact-rational auditing of a source-bound evaluation."""
if not __debug__:
    raise RuntimeError('Evaluation requires Python assertions; -O/-OO and PYTHONOPTIMIZE are forbidden')
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse, hashlib, json, math, os, re, signal, subprocess, time

ROOT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numeric(v):return type(v)in (int,float)and math.isfinite(v)
def read(path): return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def fraction(value):return {'numerator':str(value.numerator),'denominator':str(value.denominator),'value':float(value)}
def percentile(values,p):
    assert values and all(numeric(v) and v>=0 for v in values)
    return sorted(values)[math.ceil(len(values)*p)-1]
def verify_inputs(protocol):
    assert sha(__file__)==protocol['runner_sha256']
    for path,h in protocol['input_files_sha256'].items():assert sha(path)==h,path
    for path,h in protocol['implementation']['source_files_sha256'].items():assert sha(path)==h,path
    assert sha(protocol['implementation']['program'])==protocol['implementation']['program_sha256']
    assert sha(protocol['sqlite']['program'])==protocol['sqlite']['sha256']
    assert sha(protocol['sqlite']['extension'])==protocol['sqlite']['extension_sha256']
    return read(protocol['grid_path'])['tasks']
def labels(protocol,tasks):
    rows=[json.loads(line)for line in Path(protocol['dataset_directory'],'qrels.jsonl').read_text().splitlines()]
    pairs={(r['query_id'],r['chunk_external_id'])for r in rows}
    assert len(pairs)==len(rows)==3960 and sum(r['relevance']for r in rows)==91 and all(type(r['relevance'])is int and r['relevance']in (0,1)for r in rows)
    positive={t['query_id']:{r['chunk_external_id']for r in rows if r['query_id']==t['query_id']and r['relevance']==1}for t in tasks}
    eligible={t['query_id']:{r['chunk_external_id']for r in rows if r['query_id']==t['query_id']}for t in tasks}
    assert all(positive.values())
    return positive,eligible
def quality(run,tasks,positive,eligible=None):
    rows=run['per_query'];ids=[r['query_id']for r in rows]
    assert ids==[t['query_id']for t in tasks] and len(set(ids))==48
    per=[]
    for row,task in zip(rows,tasks):
        ranked=row['ranked_chunk_ids'];scores=row['scores']
        assert len(ranked)==len(set(ranked))==len(scores)<=10
        assert all(numeric(s)for s in scores)
        if eligible is not None:assert set(ranked)<=eligible[row['query_id']]
        relevant=positive[row['query_id']];rank=next((i+1 for i,id in enumerate(ranked)if id in relevant),None)
        per.append({'query_id':row['query_id'],'cell':(task['material'],task['query_type']), 'recall_at_5':Fraction(sum(id in relevant for id in ranked[:5]),len(relevant)), 'recall_at_10':Fraction(sum(id in relevant for id in ranked),len(relevant)), 'mrr_at_10':Fraction(1,rank)if rank else Fraction(0), 'first_relevant_rank':rank})
    def mean(rows,key):return sum((r[key]for r in rows),Fraction(0))/len(rows)
    overall={key:mean(per,key)for key in ['recall_at_5','recall_at_10','mrr_at_10']}
    hits=lambda rows:{'hit_at_1':Fraction(sum(r['first_relevant_rank']==1 for r in rows),len(rows)),'hit_at_2':Fraction(sum(r['first_relevant_rank']is not None and r['first_relevant_rank']<=2 for r in rows),len(rows))}
    for key,publickey in [('recall_at_5','recall_at_5'),('recall_at_10','recall_at_10'),('mrr_at_10','mrr')]:assert abs(float(overall[key])-run['metrics'][publickey])<=1e-12
    cells=[]
    for cell in sorted(set(r['cell']for r in per)):
        subset=[r for r in per if r['cell']==cell];assert len(subset)==4
        cells.append({'cell':list(cell),'tasks':4,**{key:mean(subset,key)for key in overall},'hit_diagnostics':hits(subset)})
    assert len(cells)==12
    return {'overall':overall,'cells':cells,'per_query':per,'hit_diagnostics':hits(per)}
def comparison(hybrid,baseline):
    keys=['recall_at_5','recall_at_10','mrr_at_10']
    return {'delta':{key:hybrid['overall'][key]-baseline['overall'][key]for key in keys},
            'cells':[{'cell':cell['cell'],'delta':{key:cell[key]-peer[key]for key in keys}}
                     for cell,peer in zip(hybrid['cells'],baseline['cells'])]}
def pooled_latency(runs):
    groups=[]
    for key in sorted({(r['profile'],r['mode'],r['cache'])for r in runs}):
        peers=[r for r in runs if (r['profile'],r['mode'],r['cache'])==key]
        assert len(peers)==2
        values=[v for r in peers for v in r['latency_samples_ms']];assert len(values)==96
        groups.append({'profile':key[0],'mode':key[1],'cache':key[2],'samples':len(values),
                       'p50_ms':percentile(values,.5),'p95_ms':percentile(values,.95),
                       'method':'nearest rank over unrounded pooled samples; not the mean of round percentiles',
                       'independent_tasks':48})
    return groups
def serial(value):
    if isinstance(value,Fraction):return fraction(value)
    if isinstance(value,dict):return {k:serial(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v)for v in value]
    return value
COUNTERS=['text_document_visits','text_posting_visits','vector_checks','distance_components','bm25_score_reuse','vector_score_reuse','zero_match_proof_reuse','normalization_score_visits','fusion_candidate_visits','primary_requested_bytes','index_requested_bytes','primary_read_calls','index_read_calls','primary_read_bytes','index_read_bytes']
STAGES=['preparation','recall','score_completion','normalization_fusion','result_materialization']
def integer(value):
    if type(value)is int:assert 0<=value<=9007199254740991;return value
    assert isinstance(value,str)and re.fullmatch('[1-9][0-9]*',value)
    n=int(value);assert 9007199254740991<n<=18446744073709551615;return n
def sample_plan(protocol,item):
    tasks=read(protocol['grid_path'])['tasks']
    return [{'sequence':i,'attempt':1,'round':1,'query_mode':mode,'cache_state':cache,'query_id':task['query_id'],'external_round':item['round'],'profile':item['profile']}for i,(mode,cache,task)in enumerate((m,c,t)for m in ['text_bm25','vector_exact','hybrid']for c in ['cold','warm']for t in tasks)]
def cost_verify(cost,row,profile,mode):
    assert cost['version']=='cofferdb-search-cost-v1'and cost['counter_encoding']=='exact JSON integers through 2^53-1; unsigned decimal strings above it'
    assert cost['counter_saturated']is False and cost['physical_read_bytes']is None and isinstance(cost['physical_read_reason'],str)and cost['physical_read_reason']
    assert type(cost['staging_read_bytes'])is int and cost['staging_read_bytes']==0
    assert [s['stage']for s in cost['stages']]==STAGES and set(cost['totals'])==set(COUNTERS)
    for stage in cost['stages']:
        assert type(stage['entered'])is bool and numeric(stage['elapsed_ms'])and stage['elapsed_ms']>=0
        assert set(stage['work'])==set(COUNTERS)
    for counter in COUNTERS:assert sum(integer(s['work'][counter])for s in cost['stages'])==integer(cost['totals'][counter])
    counts=cost['candidate_counts'];assert set(counts)=={'bm25_recall_count','vector_recall_count','recall_union_count','fusion_retained_count','returned_count'}
    b,v,u,m,n=[integer(counts[k])for k in ['bm25_recall_count','vector_recall_count','recall_union_count','fusion_retained_count','returned_count']]
    limit={'expanded':500,'balanced':200,'constrained':50}[profile]
    assert b<=limit and v<=limit and max(b,v)<=u<=b+v and m==min(u,limit)and n==len(row['ranked_chunk_ids'])and n<=min(10,m)
    if mode=='text_bm25':assert v==0 and u==b
    if mode=='vector_exact':assert b==0 and u==v
    assert m==row['execution_stats']['fusion_candidate_count']
    assert n==min(10,m)
    assert integer(cost['totals']['vector_checks'])>=row['execution_stats']['vector_scan_count']
    assert integer(cost['totals']['distance_components'])>=row['execution_stats']['distance_ops']
    # Trace work includes failed/reclaimed internal attempts; never equate it to
    # public work of the final successful result or discard those increments.
def context_verify(actual,protocol,profile):
    assert actual['profile']==profile.replace('-','_') and actual['profile_params']==protocol['profiles'][profile.replace('-','_')]
    assert actual['engine_params']['fusion']['algorithm']=='weighted_sum'
    assert actual['engine_params']['vector_index']['distance_metric']=='cosine'and actual['engine_params']['vector_index']['vectors_normalized']is False
    manifest=read(Path(protocol['dataset_directory'])/'dataset_manifest.json');tokenizer=manifest['tokenizer'];dataset=actual['dataset']
    assert dataset['dataset_manifest_sha256']==sha(Path(protocol['dataset_directory'])/'dataset_manifest.json')
    for key,h in manifest['hashes'].items():assert dataset[key]==h
    for key in ['dataset_id','version','chunking','tokenizer','embedding_model','query_embedding_model','query_embedding_dim','query_embedding_source']:assert dataset[key]==manifest[key]
    assert dataset['embedding_dim']==384
    assert actual['durability']=='commit'and actual['import_context']['durability']=='commit'and actual['import_context']['index_build_mode']=='inline_full_checkpoint'
    assert actual['metrics']['import_time_ms']==actual['import_context']['import_time_ms']and numeric(actual['metrics']['import_time_ms'])and actual['metrics']['import_time_ms']>=0
    summary=actual['import_context']['summary'];assert summary['documents_imported']==24 and summary['chunks_imported']==99 and summary['embeddings_imported']==99 and summary['errors']==0 and summary['model_mismatches']==0
    tasks=read(protocol['grid_path'])['tasks'];limit={'expanded':500,'balanced':200,'constrained':50}[profile]
    for run in actual['runs']:
        assert [row['query_id']for row in run['per_query']]==[task['query_id']for task in tasks]
        mode=run['query_mode'];weights={'bm25':1,'vector':0,'metadata':0}if mode=='text_bm25'else({'bm25':0,'vector':1,'metadata':0}if mode=='vector_exact'else{'bm25':.5,'vector':.5,'metadata':0})
        for row,task in zip(run['per_query'],tasks):
            context=row['query_context'];assert context['profile']==profile.replace('-','_') and context['top_k']==10 and context['candidate_limits']=={'bm25':limit,'vector':limit,'fusion':limit}
            assert context['effective_weights']==weights and context['requested_weights']==weights and context['weights_source']=='default_no_boost'and context['weights']['present']is False
            assert context['text_present']==(mode!='vector_exact')and context['embedding_present']==(mode!='text_bm25')and context['filter_present']==bool(task['filter'])
            assert context['metadata_boost_enabled']is False and context['metadata_boost_present']is False and context['boost_rule_count']==0 and context['boost_config_hash']is None
            assert context['tokenizer']==(None if mode=='vector_exact'else tokenizer) and context['index_source']=='persistent_checkpoint' and context['execution_stats']==row['execution_stats']
            assert context['transient_rebuild']['started']is False and context['transient_rebuild']['completed']is False
            assert set(row['execution_stats'])=={'vector_scan_count','distance_ops','fusion_candidate_count'}
            for v in row['execution_stats'].values():assert type(v)is int and v>=0
        for key in ['vector_scan_count','distance_ops','fusion_candidate_count']:assert run['metrics'][key]==sum(row['execution_stats'][key]for row in run['per_query'])
    assert actual['baseline_status']=='completed'and len(actual['baseline_results'])==6 and len(actual['baseline_configs'])>=1
    assert [(r['query_mode'],r['cache_state'])for r in actual['baseline_results']]==[(m,c)for m in ['text_bm25','vector_exact','hybrid']for c in ['cold','warm']]
    for baseline in actual['baseline_results']:
        assert [row['query_id']for row in baseline['per_query']]==[task['query_id']for task in tasks]and baseline['engine']=='sqlite'and baseline['name']=='sqlite-fts5-sqlite-vec'
        mode=baseline['query_mode'];expected_limits={'bm25':limit if mode!='vector_exact'else 0,'vector':limit if mode!='text_bm25'else 0,'fusion':limit if mode!='text_bm25'else 0}
        index={'text_bm25':'sqlite_fts5','vector_exact':'sqlite_vec','hybrid':'sqlite_fts5_sqlite_vec'}[mode]
        for row,task in zip(baseline['per_query'],tasks):
            context=row['query_context'];assert context['top_k']==10 and context['candidate_limits']==expected_limits and context['index_source']==index and context['tokenizer']is None
            assert context['execution_stats']==row['execution_stats']and context['filter_present']==bool(task['filter'])
            assert len(row['ranked_chunk_ids'])==len(row['scores'])<=10 and len(set(row['ranked_chunk_ids']))==len(row['ranked_chunk_ids'])and all(numeric(s)for s in row['scores'])
            assert numeric(row['latency_ms'])and row['latency_ms']>=0
        for key in ['vector_scan_count','distance_ops','fusion_candidate_count']:assert baseline['metrics'][key]==sum(row['execution_stats'][key]for row in baseline['per_query'])
        assert abs(baseline['metrics']['p95_ms']-percentile([row['latency_ms']for row in baseline['per_query']],.95))<=1e-12
    config=actual['baseline_configs'][0];assert config['version']==protocol['sqlite']['version']and config['config']['vector_extension_version']=='v'+protocol['sqlite']['extension_version']and config['config']['query_count']==48
    assert config['name']=='sqlite-fts5-sqlite-vec'and config['cache_states']==['cold','warm']and config['requires_query_embedding']is True
    assert all(config['config'][key]==value for key,value in {'engine':'sqlite','fts':'fts5','vector_mode':'exact_scan','fusion':'weighted_sum','status':'completed'}.items())
    # Baseline configuration/strategy and complete result are retained as-is;
    # CofferDB profile candidate policy is not imposed on SQLite.
def sqlite_sample_projection(directory,protocol,item,verified=False,reason=None):
    plan=sample_plan(protocol,item)
    if verified:
        result=read(directory/'result.json')
        actual=[(r,row)for r in result['baseline_results']for row in r['per_query']]
        assert len(actual)==len(plan)==288
        samples=[]
        for planned,(run,row)in zip(plan,actual):
            assert planned['query_id']==row['query_id']and planned['query_mode']==run['query_mode']and planned['cache_state']==run['cache_state']
            samples.append({**planned,'engine':'sqlite','status':'completed','latency_ms':row['latency_ms'],'reason':None})
        return {'samples':samples,'evidence':'Complete result plus all frozen context/metric/journal assertions passed.',
                'result_sha256':sha(directory/'result.json'),'individual_start_events_available':False}
    return {'samples':[{**p,'engine':'sqlite','status':'unknown_not_observed','latency_ms':None,
                        'reason':reason or 'Launched profile has no fully verified baseline result; individual baseline progress is unavailable.'}for p in plan],
            'evidence':'No per-sample SQLite terminal journal; do not infer success or which baseline query ran.',
            'individual_start_events_available':False,
            'raw_logs_sha256':{p.name:sha(p)for p in [directory/'stdout',directory/'stderr']if p.exists()}}

def project_attempt(directory,protocol,item):
    plan=sample_plan(protocol,item);records=[dict(p,status='not_run',search_cost_snapshot=None,terminal_elapsed_ms=None,error_code=None,reason=None)for p in plan]
    files=list(directory.glob('cofferdb-benchmark-execution-*.jsonl'));info={'raw_journals_sha256':{str(p):sha(p)for p in files},'raw_logs_sha256':{p.name:sha(p)for p in [directory/'stdout',directory/'stderr']if p.exists()},'samples':records,'malformed_events':[],'journal_absent':not files}
    if len(files)!=1:
        for record in records:record.update(status='unknown_not_observed',reason='No unique readable execution journal; individual query progress is unknown.')
        return info
    raw=files[0].read_bytes();lines=raw.split(b'\n');info['trailing_unterminated_bytes']=len(lines[-1]);info['unterminated_prefix_sha256']=hashlib.sha256(lines[-1]).hexdigest()if lines[-1]else None
    for number,line in enumerate(lines[:-1]):
        try:
            event=json.loads(line);kind=event.get('event')
            if kind not in ('planned','started','search_completed','search_failed','completed','failed','not_run'):continue
            i=event['sequence'];assert type(i)is int and 0<=i<288;record=records[i]
            assert all(event[k]==record[k]for k in ['sequence','attempt','round','query_mode','cache_state','query_id'])
            if kind=='started':record['status']='incomplete_started_no_terminal'
            elif kind in ('completed','failed','not_run'):
                record.update(status=kind,terminal_elapsed_ms=event.get('elapsed_ms'),error_code=event.get('error_code'),reason=event.get('reason'))
            if kind in ('search_completed','search_failed'):
                assert 'search_event'not in record;record['search_cost_snapshot']=event['search_cost'];record['search_event']=kind;record['search_elapsed_ms']=event['elapsed_ms']
        except BaseException as error:info['malformed_events'].append({'line':number+1,'error':repr(error)})
    if (directory/'supervisor-outcome.json').exists():
        supervisor=read(directory/'supervisor-outcome.json');info['supervisor_outcome_sha256']=sha(directory/'supervisor-outcome.json')
        for sample in supervisor['active_samples_at_exit']:
            record=records[sample['sequence']]
            if record['status']=='incomplete_started_no_terminal':
                record.update(observed_duration_seconds=sample['observed_duration_seconds'],observation_kind='elapsed since supervisor first observed flushed started event; lower bound, not completed search latency',supervisor_timeout_reason=supervisor['timeout_reason'],supervisor_error=supervisor['supervisor_error'])
    stop_reason='Profile stopped before a terminal sample was observed; see supervisor outcome and raw logs.'
    if (directory/'supervisor-outcome.json').exists():
        outcome=read(directory/'supervisor-outcome.json')
        stop_reason=outcome.get('timeout_reason') or outcome.get('supervisor_error') or ('profile_exit_'+str(outcome.get('exit')))
    for record in records:
        if record['status']=='not_run'and (info['malformed_events']or info['trailing_unterminated_bytes']):
            record.update(status='unknown_not_observed',reason='Malformed or unterminated journal prevents proving this sample was unexecuted.')
        elif record['status']=='not_run'and not record['reason']:record['reason']=stop_reason
    return info
def journal_verify(directory,protocol,result,profile,item):
    files=list(directory.glob('cofferdb-benchmark-execution-*.jsonl'));assert len(files)==1
    path=files[0];events=[json.loads(line)for line in path.read_text().splitlines()]
    header=events[0];last=events[-1]
    assert header['rules_version']=='cofferdb-builtin-source-acceptance-v4' and header['search_cost_version']=='cofferdb-search-cost-v1'
    assert header['git_commit']==protocol['implementation']['commit'] and header['profile']==profile.replace('-','_') and header['planned_samples']==288
    assert last['event']=='run_completed' and last['completed_samples']==288 and last['result_pretty_json_sha256']==sha(result)
    assert last['import_time_ms']==read(result)['metrics']['import_time_ms']==read(result)['import_context']['import_time_ms']
    assert header['durability']=='commit'and header['warmup_rounds']==0 and header['cold_reset']=='reopen handle; no OS page-cache eviction'
    counts=Counter(e['event']for e in events)
    assert all(counts[name]==288 for name in ['planned','started','search_completed','completed'])
    assert all(counts[name]==0 for name in ['failed','not_run','run_failed','search_failed'])
    plan=sample_plan(protocol,item);actual=read(result);rows=[row for run in actual['runs']for row in run['per_query']];assert len(rows)==288
    ordered=[(e['event'],e['sequence'])for e in events if e['event']in ('started','search_completed','completed')]
    assert ordered==[(kind,i)for i in range(288)for kind in ['started','search_completed','completed']]
    expected_events=[e for e in events if e['event']=='expected'];assert len(expected_events)==144
    expected={(e['query_mode'],e['query_id']):e for e in expected_events}
    tasks=read(protocol['grid_path'])['tasks'];assert len(expected)==144 and set(expected)=={(mode,t['query_id'])for mode in ['text_bm25','vector_exact','hybrid']for t in tasks}
    for run in actual['runs']:
        for row in run['per_query']:
            oracle=expected[(run['query_mode'],row['query_id'])]
            assert row['ranked_chunk_ids']==oracle['ranked_chunk_ids']and len(row['scores'])==len(oracle['scores'])
            assert all(numeric(a)and numeric(b)and abs(a-b)<=1e-12*max(1,abs(b))for a,b in zip(row['scores'],oracle['scores']))
    costs=[]
    by_event={name:{(e['sequence'],e['attempt']):e for e in events if e['event']==name}for name in ['planned','started','search_completed','completed']}
    assert all(len(group)==288 for group in by_event.values())
    assert all(set(group)=={(i,1)for i in range(288)}for group in by_event.values())
    for name,group in by_event.items():
        assert [e['sequence']for e in events if e['event']==name]==list(range(288))
        for key,event in group.items():assert all(event[k]==plan[key[0]][k]for k in ['sequence','attempt','round','query_mode','cache_state','query_id'])
    for key,e in by_event['search_completed'].items():
        cost=e['search_cost'];terminal=by_event['completed'][key]
        assert terminal['search_cost']==cost and terminal['elapsed_ms']==e['elapsed_ms']
        row=rows[key[0]];assert row['query_id']==e['query_id']and row['latency_ms']==e['elapsed_ms'] and terminal['index_source']==row['query_context']['index_source']
        cost_verify(cost,row,profile,e['query_mode'])
        assert math.isfinite(e['elapsed_ms'])and e['elapsed_ms']>=0
        if e['elapsed_ms']>120000:raise RuntimeError('query_attempt_deadline_120s_in_terminal_journal')
        costs.append({'sequence':e['sequence'],'attempt':e['attempt'],'query_id':e['query_id'],'query_mode':e['query_mode'],'cache_state':e['cache_state'],'elapsed_ms':e['elapsed_ms'],'search_cost':cost})
    return {'path':str(path),'sha256':sha(path),'events':dict(counts),'search_costs_counted_once':costs}
def supervise(command,directory,protocol):
    environment=os.environ.copy();environment.update(COFFERDB_SQLITE3=protocol['sqlite']['program'],COFFERDB_SQLITE_VEC_EXTENSION=protocol['sqlite']['extension'],TMPDIR=str(directory))
    started=time.monotonic();active_queries={};finished=set();offset=0;buffer=b'';journal=None;reason=None;process=None;error=None;timed_sample=None;leader_exit_seen=False
    try:
        with (directory/'stdout').open('xb')as out,(directory/'stderr').open('xb')as err:
            process=subprocess.Popen(['/usr/bin/time','-l',*command],cwd=protocol['implementation']['source_directory'],env=environment,stdout=out,stderr=err,start_new_session=True)
            while True:
                if journal is None:
                    candidates=list(directory.glob('cofferdb-benchmark-execution-*.jsonl'));assert len(candidates)<=1
                    if candidates:journal=candidates[0]
                if journal is not None:
                    with journal.open('rb')as stream:stream.seek(offset);data=stream.read();offset+=len(data)
                    buffer+=data;lines=buffer.split(b'\n');buffer=lines.pop()
                    for line in lines:
                        event=json.loads(line);kind=event.get('event');key=(event.get('sequence'),event.get('attempt'))
                        if kind=='started':
                            assert key not in active_queries and key not in finished;active_queries[key]={'observed_start':time.monotonic(),'query_id':event['query_id'],'query_mode':event['query_mode'],'cache_state':event['cache_state']}
                        elif kind in ('completed','failed'):
                            assert key in active_queries;sample=active_queries[key];duration=time.monotonic()-sample['observed_start']
                            if duration>120 or isinstance(event.get('elapsed_ms'),(int,float))and event['elapsed_ms']>120000:
                                reason='query_attempt_deadline_120s';timed_sample={'sequence':key[0],'attempt':key[1],**sample,'observed_duration_seconds':duration,'reported_search_elapsed_ms':event.get('elapsed_ms')}
                            active_queries.pop(key);finished.add(key)
                for key,sample in active_queries.items():
                    if time.monotonic()-sample['observed_start']>120:
                        reason='query_attempt_deadline_120s';timed_sample={'sequence':key[0],'attempt':key[1],**sample,'observed_duration_seconds':time.monotonic()-sample['observed_start']};break
                if time.monotonic()-started>7200:reason='profile_deadline_7200s'
                if reason:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait();break
                if process.poll()is not None:
                    # Drain once more after observing exit: the final flushed event
                    # can arrive between the preceding read and poll.
                    if leader_exit_seen:break
                    leader_exit_seen=True
                    continue
                time.sleep(.02)
    except BaseException as failure:
        error=repr(failure)
        if process is not None and process.poll()is None:
            try:os.killpg(process.pid,signal.SIGKILL);process.wait()
            except BaseException as cleanup:error+='; cleanup='+repr(cleanup)
    absent=process is None;group_observations=[];cleanup_signals=[]
    if process is not None:
        # A reaped leader may leave descendants in its owned session/process group.
        # Clean that group even when process.poll() already reports an exit.
        try:
            os.killpg(process.pid,0)
            error=(error or '')+'; owned_group_remained_after_leader_exit_or_failure'
            cleanup_signals.append('SIGKILL_owned_group_after_leader_exit_or_supervisor_failure')
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            absent=True
        except BaseException as cleanup:
            error=(error or '')+'; owned_group_cleanup='+repr(cleanup)
        group_deadline=time.monotonic()+2
        while not absent:
            try:os.killpg(process.pid,0);observation='exists'
            except ProcessLookupError:absent=True;group_observations.append('absent');break
            except BaseException as check:observation=repr(check)
            group_observations.append(observation)
            if time.monotonic()>=group_deadline:
                error=(error or '')+'; process_group_not_proven_absent='+observation;break
            time.sleep(.02)
    outcome={'exit':process.returncode if process is not None else None,'owned_process_group':process.pid if process is not None else None,'group_check_observations':group_observations,'cleanup_signals':cleanup_signals,'supervisor_error':error,'timeout_reason':reason,'timeout_sample':timed_sample,'elapsed_seconds':time.monotonic()-started,'owned_process_group_absent':absent,'active_samples_at_exit':[{'sequence':key[0],'attempt':key[1],**sample,'observed_duration_seconds':time.monotonic()-sample['observed_start']}for key,sample in active_queries.items()],'raw_logs_sha256':{p.name:sha(p)for p in [directory/'stdout',directory/'stderr']if p.exists()},'supervision_limitations':'Deadlines are external observed journal durations plus reported search latency;20ms polling and host scheduling may delay observation/termination. Not a strict internal engine deadline.'}
    save(directory/'supervisor-outcome.json',outcome)
    return outcome
def execute(protocol_path,approval_path,output):
    protocol=read(protocol_path);approval=read(approval_path)
    assert approval['protocol_sha256']==sha(protocol_path)and approval['product_approved']is True and approval['test_review_passed']is True
    tasks=verify_inputs(protocol);positive,eligible=labels(protocol,tasks);output.mkdir();attempts=[]
    record={'status':'running','protocol_sha256':sha(protocol_path),'approval_sha256':sha(approval_path),'attempts':attempts,'not_run':protocol['execution_plan'].copy(),'full_sample_plan':protocol['full_sample_plan'],'planned_cofferdb_samples':1728,'planned_sqlite_samples':1728,'full_sqlite_sample_plan':protocol['full_sqlite_sample_plan']}
    def update():
        temporary=output/'execution.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(output/'execution.json')
    update();quality_runs=[];signatures=[];external_baselines=[]
    try:
        for item in protocol['execution_plan']:
            verify_inputs(protocol);record['not_run'].pop(0);attempt=dict(item,status='running');attempts.append(attempt);update()
            directory=output/f"round-{item['round']}-{item['profile']}";directory.mkdir();result=directory/'result.json'
            command=[protocol['implementation']['program'],'benchmark',str(directory/'data.coffer'),'--dataset',protocol['dataset_directory'],'--collection','benchmark','--profile',item['profile'],'--durability','commit','--result-file',str(result),'--require-baseline']
            attempt['command']=command;attempt.update(supervise(command,directory,protocol));update()
            assert attempt['exit']==0 and attempt['timeout_reason']is None and attempt['supervisor_error']is None and attempt['owned_process_group_absent']is True
            actual=read(result);assert actual['git_commit']==protocol['implementation']['commit']and actual['baseline_status']=='completed'and actual['index_build_mode']=='inline_full_checkpoint'
            assert [(r['query_mode'],r['cache_state'])for r in actual['runs']]==[(m,c)for m in ['text_bm25','vector_exact','hybrid']for c in ['cold','warm']]
            context_verify(actual,protocol,item['profile'])
            attempt['result_sha256']=sha(result);attempt['journal']=journal_verify(directory,protocol,result,item['profile'],item)
            rss=re.search(r'(\d+)\s+maximum resident set size',(directory/'stderr').read_text());assert rss;attempt['whole_process_peak_rss_bytes']=int(rss[1])
            for run in actual['runs']:
                q=quality(run,tasks,positive,eligible)
                q.update(round=item['round'],profile=item['profile'],mode=run['query_mode'],cache=run['cache_state'],p50_ms=percentile([x['latency_ms']for x in run['per_query']],.5),p95_ms=percentile([x['latency_ms']for x in run['per_query']],.95))
                assert abs(q['p95_ms']-run['metrics']['p95_ms'])<=1e-12
                q['latency_samples_ms']=[x['latency_ms']for x in run['per_query']]
                quality_runs.append(q)
                signatures.append({'round':item['round'],'profile':item['profile'],'mode':run['query_mode'],'cache':run['cache_state'],'rows':[(r['query_id'],r['ranked_chunk_ids'],r['scores'])for r in run['per_query']]})
            for run in actual['baseline_results']:
                diagnostic=quality(run,tasks,positive,eligible);diagnostic.update(round=item['round'],profile=item['profile'],mode=run['query_mode'],cache=run['cache_state'],p50_ms=percentile([x['latency_ms']for x in run['per_query']],.5),p95_ms=percentile([x['latency_ms']for x in run['per_query']],.95),scope='externalSQLite diagnostics; separate engine strategy; startup-inclusive timing and own candidate/filter/fusion strategy')
                diagnostic['latency_samples_ms']=[x['latency_ms']for x in run['per_query']]
                external_baselines.append(diagnostic)
            verify_inputs(protocol)
            attempt['sample_projection']=project_attempt(directory,protocol,item);assert all(s['status']=='completed'for s in attempt['sample_projection']['samples'])and not attempt['sample_projection']['malformed_events']
            attempt['sqlite_sample_projection']=sqlite_sample_projection(directory,protocol,item,verified=True)
            attempt['status']='passed';update()
        assert len(quality_runs)==36
        hybrid=[r for r in quality_runs if r['mode']=='hybrid'];assert len(hybrid)==12
        comparisons=[]
        for r in hybrid:
            if r['mode']=='hybrid':
                for mode in ['text_bm25','vector_exact']:
                    base=next(b for b in quality_runs if b['round']==r['round']and b['profile']==r['profile']and b['cache']==r['cache']and b['mode']==mode)
                    comparisons.append({'round':r['round'],'profile':r['profile'],'cache':r['cache'],'baseline':mode,**comparison(r,base)})
        for signature in signatures:
            assert all(signature['rows']==peer['rows']for peer in signatures if peer['profile']==signature['profile']and peer['mode']==signature['mode'])
        for r in quality_runs:
            peer=next(b for b in quality_runs if b['round']==3-r['round']and b['profile']==r['profile']and b['mode']==r['mode']and b['cache']==r['cache'])
            assert r['overall']==peer['overall']and r['cells']==peer['cells']
        report={'execution_complete':True,'formal_queries':1728,'external_baseline_queries':1728,'independent_tasks':48,'quality_runs':quality_runs,'external_baseline_diagnostics':external_baselines,'hybrid_single_branch_comparisons':comparisons,'pooled_latency':{'cofferdb':pooled_latency(quality_runs),'sqlite':pooled_latency(external_baselines)},'evaluation_completeness':'complete','scheme_certification':'not_requested','engine_contract_observation':'all frozen contextual, source-reference and journal assertions passed; not full version acceptance','limitations':protocol['limitations']}
        save(output/'quality-report.json',serial(report));record.update(status='passed',evaluation_completeness='complete',scheme_certification='not_requested',quality_report_sha256=sha(output/'quality-report.json'))
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        if attempts:
            attempt=attempts[-1];directory=output/f"round-{attempt['round']}-{attempt['profile']}"
            attempt['sample_projection']=project_attempt(directory,protocol,attempt)
            if attempt['status']=='running':attempt['status']='failed'
            if 'sqlite_sample_projection'not in attempt:
                attempt['sqlite_sample_projection']=sqlite_sample_projection(directory,protocol,attempt,reason=record['error'])
        record['not_run_sample_plans']=[[{**p,'status':'not_run','reason':'Aborted after earlier profile failure: '+record['error']}for p in sample_plan(protocol,item)]for item in record['not_run']]
        record['not_run_sqlite_sample_plans']=[[{**p,'engine':'sqlite','status':'not_run','latency_ms':None,'reason':'Profile was never launched after earlier failure: '+record['error']}for p in sample_plan(protocol,item)]for item in record['not_run']]
        raise
    finally:update()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('protocol',type=Path);parser.add_argument('--check-inputs',action='store_true');parser.add_argument('--approval',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check_inputs:
        protocol=read(args.protocol);tasks=verify_inputs(protocol);labels(protocol,tasks);print(json.dumps({'status':'inputs_verified_no_retrieval','tasks':len(tasks)}))
    else:
        assert args.approval and args.output
        execute(args.protocol,args.approval,args.output)
if __name__=='__main__':main()
