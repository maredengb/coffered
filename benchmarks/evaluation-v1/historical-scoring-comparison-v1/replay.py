"""Bounded replay of the historical scoring comparison; each run has a new identity."""
import selectors
import stat
import argparse,hashlib,json,math,os,platform,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILES=['quality-first','balanced','resource-first']  # Only the frozen historical engines' internal identifiers.
POLICIES=['window_scores','complete_U']

def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def summary(values):
 need(values and all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in values),'invalid times')
 a=sorted(values);return {'samples':len(a),'p50_ms':a[math.ceil(len(a)*.5)-1],'p95_ms':a[math.ceil(len(a)*.95)-1]}
def owned_logical_bytes():
 total=0
 def disappeared_only(error):
  if not isinstance(error,FileNotFoundError):raise error
 for directory,_,files in os.walk(ROOT,followlinks=False,onerror=disappeared_only):
  for name in files:
   try:info=(Path(directory)/name).lstat()
   except FileNotFoundError:continue  # Already removed at observation time.
   if stat.S_ISREG(info.st_mode):total+=info.st_size
 return total

def identity(protocol):
 need(platform.system()=='Darwin' and platform.machine()=='arm64','replay requires macOS arm64')
 need(str(ROOT)==protocol['preparation_root'],'rebuild after moving a prepared work directory')
 seen=set()
 for row in protocol['files']:
  relative=Path(row['path']);need(not relative.is_absolute() and '..' not in relative.parts and str(relative) not in seen,'invalid or duplicate bound path');seen.add(str(relative))
  p=ROOT/relative
  need(all(not parent.is_symlink() for parent in [p,*p.parents] if parent!=ROOT.parent),'symlink in bound path')
  need(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'changed replay input '+str(relative))
 need(owned_logical_bytes()<=8*1024**3,'owned preparation/results exceed 8GiB stop threshold')
 available=os.statvfs(ROOT)
 need(available.f_bavail*available.f_frsize>=28*1024**3,'free capacity less than 28GiB preparation reserve')

def storage_safe():
 available=os.statvfs(ROOT);free=available.f_bavail*available.f_frsize
 owned=owned_logical_bytes()
 return free>=20*1024**3 and owned<=8*1024**3

def run_process(argv,out,timeout,*,max_output=64*1024**2):
 """Bound capture and record stop, reap and drain separately from API query timing."""
 started_ns=time.monotonic_ns();p=None;stop_ns=None;reap_ns=None;why=None;peak=0
 paths={name:out.with_suffix('.'+name) for name in ['stdout','stderr']}
 receipt={'argv':argv,'exit':None,'stop_reason':None,'started_monotonic_ns':started_ns,
          'stop_observed_monotonic_ns':None,'reaped_monotonic_ns':None,'drain_complete':False,
          'stdout_sha256':None,'stderr_sha256':None,'stream_truncated':False,'spawned':False,'capture_integrity':True,'capture_errors':[],'created_streams':[]}
 if any(path.exists() or path.is_symlink() for path in paths.values()):
  return {**receipt,'stop_reason':'output_collision','wall_ms':0}
 env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/dev/null','LANG':'C','LC_ALL':'C','TZ':'UTC'}
 targets={};counts={name:0 for name in paths};selector=selectors.DefaultSelector()
 def consume(wait):
  nonlocal why,stop_ns
  for key,_ in selector.select(wait):
   name=key.data
   try:chunk=os.read(key.fd,65536)
   except BaseException as exc:
    receipt['capture_integrity']=False;receipt['capture_errors'].append(name+' pipe read: '+type(exc).__name__+': '+str(exc)[:512]);raise
   if not chunk:selector.unregister(key.fileobj);continue
   room=max_output-counts[name];payload=chunk[:room]
   try:
    written=targets[name].write(payload)
    if type(written) is not int or not 0<=written<=len(payload):raise OSError('invalid capture write count')
    counts[name]+=written
    if written!=len(payload):raise OSError('short capture write')
   except BaseException as exc:
    receipt['capture_integrity']=False;receipt['capture_errors'].append(name+' capture write after pipe read: '+type(exc).__name__+': '+str(exc)[:512]);raise
   if len(chunk)>room:
    receipt['stream_truncated']=True
    if why is None:why='log_limit';stop_ns=time.monotonic_ns()
 try:
  for name,path in paths.items():
   targets[name]=path.open('xb');receipt['created_streams'].append(name)
  p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  receipt['spawned']=True
  for name in paths:selector.register(getattr(p,name),selectors.EVENT_READ,name)
  next_guard=time.monotonic()+1
  while selector.get_map() or p.poll() is None:
   now=time.monotonic()
   if (time.monotonic_ns()-started_ns)/1e9>=timeout:why='timeout';stop_ns=time.monotonic_ns();break
   if now>=next_guard:
    if not storage_safe():why='observed_storage_threshold';stop_ns=time.monotonic_ns();break
    scan=subprocess.check_output(['/bin/ps','-A','-o','pgid=,rss='],env=env,timeout=5)
    value=sum(int(row.split()[1])*1024 for row in scan.splitlines() if len(row.split())==2 and int(row.split()[0])==p.pid)
    peak=max(peak,value);next_guard=now+1
    if value>4*1024**3:why='observed_group_rss_threshold';stop_ns=time.monotonic_ns();break
   consume(.05)
   if why:break
 except BaseException as exc:
  if why is None:why='process_exception';stop_ns=time.monotonic_ns()
  receipt.update(error_type=type(exc).__name__,error_message=str(exc)[:1024])
 finally:
  if p is not None:
   # poll() reaps; never signal a group after its leader has been reaped.
   if p.poll() is None:
    if stop_ns is None:stop_ns=time.monotonic_ns()
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
   try:p.wait(timeout=5);reap_ns=time.monotonic_ns();receipt['exit']=p.returncode
   except BaseException as exc:
    why=why or 'cleanup_exception';receipt['cleanup_error']=type(exc).__name__+': '+str(exc)[:512]
   drain_until=time.monotonic()+2
   try:
    while selector.get_map() and time.monotonic()<drain_until:consume(.02)
    receipt['drain_complete']=not bool(selector.get_map())
   except BaseException as exc:
    receipt['drain_error']=type(exc).__name__+': '+str(exc)[:512];why=why or 'drain_exception'
   for name in paths:
    stream=getattr(p,name,None)
    if stream is not None:
     try:stream.close()
     except BaseException as exc:receipt['capture_errors'].append(name+' pipe close: '+type(exc).__name__+': '+str(exc)[:512])
  try:selector.close()
  except BaseException as exc:receipt['capture_errors'].append('selector close: '+type(exc).__name__+': '+str(exc)[:512])
  for name,stream in targets.items():
   try:stream.close()
   except BaseException as exc:receipt['capture_errors'].append(name+' capture close: '+type(exc).__name__+': '+str(exc)[:512])
   try:receipt[name+'_sha256']=sha(paths[name])
   except BaseException as exc:receipt['capture_errors'].append(name+' hash: '+type(exc).__name__+': '+str(exc)[:512])
 if receipt['capture_errors']:
  receipt['capture_integrity']=False;why=why or 'capture_finalize_exception'
  if stop_ns is None:stop_ns=time.monotonic_ns()
 receipt.update(stop_reason=why,stop_observed_monotonic_ns=stop_ns,reaped_monotonic_ns=reap_ns,
                wall_ms=(time.monotonic_ns()-started_ns)/1e6,observed_group_rss_peak_bytes=peak,
                captured_bytes=counts)
 return receipt


def parse_rows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def rss(path):
 found=re.search(r'(\d+)\s+maximum resident set size',path.read_text())
 need(found is not None,'missing actual time -l RSS');return int(found[1])
class SampleValidationError(ValueError):
 def __init__(self,sample_id,reason):
  self.sample_id=sample_id;super().__init__(str(sample_id)+': '+str(reason))

def validate_quality(rows,role,profile,diagnostic):
 qrels={}
 for r in parse_rows(ROOT/'quality-input/qrels.jsonl'):
  key=(r['query_id'],r['chunk_external_id']);need(key not in qrels,'duplicate qrel');qrels[key]=r['relevance']
 queries=[r['query_id'] for r in parse_rows(ROOT/'quality-input/queries.jsonl')]
 expected=[(mode,q,cache) for mode in ['text','vector','hybrid'] for q in queries for cache in ['fresh_handle','warm_same_handle']]
 need(len(rows)==288,'quality sample count');metrics=[]
 for r,(mode,q,cache) in zip(rows,expected):
  try:
   need((r['mode'],r['query_id'],r['cache'],r['profile'],r['policy'])==(mode,q,cache,profile.replace('-','_'),role),'quality identity/order')
   need(math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>=0,'quality timer')
   validate_reads(r['reads'],diagnostic)
   hits=r['result']['results'];ids=[x['chunk_external_id'] for x in hits];need(len(ids)<=10 and len(set(ids))==len(ids),'top10 duplicates/length')
   need(all((q,key) in qrels for key in ids),'unjudged return')
   positives={key for (query,key),relevance in qrels.items() if query==q and relevance>=1};need(positives,'no positives')
   relevant=[i for i,key in enumerate(ids,1) if key in positives]
   metrics.append({'query_id':q,'mode':mode,'cache':cache,'recall10':len(relevant)/len(positives),'mrr10':0 if not relevant else 1/relevant[0],'elapsed_ms':r['elapsed_ms']})
  except Exception as exc:raise SampleValidationError(r.get('sample_id'),exc) from exc
 return metrics

def validate_reads(r,diagnostic):
 if not diagnostic:need(r is None,'instrumentation mixed with primary timing');return
 need(type(r) is dict and r['scope']=='querying_thread_userspace_read_pread_readv' and r['physical_read_bytes'] is None and r['saturated'] is False and r['errors']==0,'invalid read scope/error')
 for k in ['calls','known_requested_bytes','returned_bytes']:need(type(r[k]) is int and r[k]>=0,'invalid read counter')
 need(r['returned_bytes']<=r['known_requested_bytes'],'read bytes exceed request')

def actual_layers(result,role):
 """Expose actual historical semantics, never reinterpret configuration as |U|."""
 hits=result['results'];execution=result['execution_stats']
 limits=result['query_context']['candidate_limits']
 normalizations=[hit['explain']['normalization']['candidate_count'] for hit in hits]
 measured_union=normalizations[0] if role=='complete_U' and normalizations and len(set(normalizations))==1 else None
 branch=result['candidate_counts'] if role=='complete_U' else None
 return {'candidate_limits':limits,
         'branch_recalled_after_filter':{'bm25':None if branch is None else branch['bm25'],'vector':None if branch is None else branch['vector'],'status':'not_measured' if branch is None else 'measured','reason':'window_scores candidate_counts JSON aliases configuration limits; original API does not expose actual branch counts' if branch is None else 'complete_U db.rs candidate_counts assigned from filtered bm25_hits/vector_hits lengths'},
         'complete_union':{'count':measured_union,'status':'measured' if measured_union is not None else 'not_measured','reason':'complete_U explain normalization.candidate_count before fusion limit; consistent across returned rows' if measured_union is not None else 'no returned-row witness' if role=='complete_U' else 'window_scores normalization.candidate_count is post-fusion count, not pre-limit union'},
         'fusion_retained':execution['fusion_candidate_count'],'returned':len(hits),
         'normalization_count_witnesses':normalizations,
         'legacy_candidate_counts_json_semantics':'actual_branch_and_post_fusion_counts' if role=='complete_U' else 'configuration_limits_alias_not_actual_counts'}

def validate_scale_result(row,role,profile):
 try:
  result=row['result'];hits=result['results'];mode=row['mode']
  need(row['result_retained_before_validation'] is True,'scale raw result not retained before assertions')
  need(type(row['returned_count']) is int and row['returned_count']==len(hits)==10,'scale actual returned count')
  need([hit['chunk_external_id'] for hit in hits]==[f'chunk-{n}' for n in range(76400,76410)],'scale actual Top10 IDs/order')
  need([hit['rank'] for hit in hits]==list(range(1,11)) and all(type(hit['rank']) is int for hit in hits),'scale actual ranks')
  for hit in hits:
   need(type(hit['final_score']) in [int,float] and math.isfinite(hit['final_score']) and hit['final_score']==1.0,'scale actual final score')
   for key in ['bm25_score','vector_score']:
    value=hit[key];need(value is None or type(value) in [float,int] and math.isfinite(value),'scale raw score must be retained/finite')
   normalized=hit['normalized_scores']
   need(normalized['bm25']==(0 if mode=='vector' else 1),'scale normalized BM25 oracle')
   need(normalized['vector']==(None if mode=='text' else 1),'scale normalized vector oracle')
   need(normalized['metadata_boost']==0,'scale normalized metadata oracle')
  layers=actual_layers(result,role);cap={'quality-first':500,'balanced':200,'resource-first':50}[profile]
  need(layers['candidate_limits']==dict(bm25=cap,vector=cap,fusion=cap),'scale raw result limits')
  need(type(layers['fusion_retained']) is int and 10<=layers['fusion_retained']<=cap,'scale actual fusion retained')
  need(result['profile']==profile.replace('-','_') and result['top_k']==10,'scale raw profile/top_k')
  need(result['execution_stats']==dict(vector_scan_count=row['vector_scan_count'],distance_ops=row['distance_ops'],fusion_candidate_count=row['fusion_candidate_count']),'scale raw work counters')
  if role=='complete_U':
   branch=layers['branch_recalled_after_filter'];need(all(type(branch[k]) is int and 0<=branch[k]<=cap for k in ['bm25','vector']),'scale actual branch counts')
   union=layers['complete_union']['count'];need(type(union) is int and max(branch['bm25'],branch['vector'])<=union<=branch['bm25']+branch['vector'],'scale measured U witnesses/bounds')
  need(result['query_context']['index_source']==row['index_source'],'scale raw index source')
  return layers
 except Exception as exc:raise SampleValidationError(row.get('sample_id'),exc) from exc

def validate_scale(rows,role,profile,diagnostic):
 phases=[('first_query',1),('warmup',0 if diagnostic else 2),('measured',1 if diagnostic else 20)]
 expected_samples=16 if diagnostic else 184;need(len(rows)==expected_samples+9,'scale row count')
 need(rows[-1]=={'stage':'complete','groups':8,'planned_query_samples':expected_samples,'verified':True},'scale complete')
 index=0;groups=[];cap={'quality-first':500,'balanced':200,'resource-first':50}[profile]
 for mode in ['text','vector','hybrid','hybrid-filtered']:
  for scope in ['rust_search_execute','rust_search_standard_json']:
   opened=rows[index];index+=1;need((opened['stage'],opened['profile'],opened['mode'],opened['scope'])==('open',profile.replace('-','_'),mode,scope),'scale group')
   stats=opened['collection'];need((stats['document_count'],stats['chunk_count'],stats['embedding_count'])==(1000,100000,100000),'scale input counts')
   group=[]
   for phase,count in phases:
    for sample in range(1,count+1):
     r=rows[index];index+=1
     try:
      need((r['stage'],r['profile'],r['mode'],r['scope'],r['phase'],r['round'],r['result_retained_before_validation'])==('query',profile.replace('-','_'),mode,scope,phase,sample,True),'scale sample order')
      r['candidate_layers']=validate_scale_result(r,role,profile)
      scored=0 if mode=='text' else 5000 if mode=='hybrid-filtered' else 100000
      scan=100000 if mode=='hybrid-filtered' and role=='complete_U' else scored
      need(r['vector_scan_count']==scan and r['distance_ops']==scored*768,'scale scan/distance')
      need(r['candidate_limits']==dict(bm25=cap,vector=cap,fusion=cap) and r['index_source']=='persistent_checkpoint','scale profile/index')
      validate_reads(r['reads'],diagnostic);need(math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>=0,'scale time')
      need((r['serialized_bytes']==0) if scope=='rust_search_execute' else (type(r['serialized_bytes']) is int and r['serialized_bytes']>0),'scale JSON boundary')
      group.append(r)
     except Exception as exc:raise SampleValidationError(r.get('sample_id'),exc) from exc
   groups.append({'mode':mode,'scope':scope,'primary_latency':None if diagnostic else summary([r['elapsed_ms'] for r in group if r['phase']=='measured']),'all_samples':group})
 return groups

def plan():
 result=[]
 for scope in ['quality','scale']:
  for diagnostic in [False,True]:
   for round_ in ([1] if diagnostic else [1,2]):
    for profile in PROFILES:
     for role in (POLICIES if round_==1 else list(reversed(POLICIES))):result.append({'scope':scope,'diagnostic':diagnostic,'round':round_,'profile':profile,'policy':role})
 return result

def sample_plan(item,trial_id):
 common={**item,'trial_id':trial_id,'attempt':1}
 rows=[]
 if item['scope']=='quality':
  for mode in ['text','vector','hybrid']:
   for query in parse_rows(ROOT/'quality-input/queries.jsonl'):
    q=query['query_id']
    for cache in ['fresh_handle','warm_same_handle']:
     rows.append({**common,'sample_id':f'quality:{mode}:{q}:{cache}','query_id':q,'mode':mode,'cache':cache})
 else:
  ids={'text':'scale-text-764','vector':'scale-vector-e0','hybrid':'scale-hybrid-764-e0','hybrid-filtered':'scale-hybrid-filtered-764-e0-project4'}
  phases=[('first_query',1),('warmup',0 if item['diagnostic'] else 2),('measured',1 if item['diagnostic'] else 20)]
  for mode,q in ids.items():
   for scope in ['rust_search_execute','rust_search_standard_json']:
    for phase,count in phases:
     for iteration in range(1,count+1):
      rows.append({**common,'sample_id':f'scale:{mode}:{scope}:{phase}:{iteration}','query_id':q,'mode':mode,'api_scope':scope,'phase':phase,'iteration':iteration})
 return rows

def not_run_sample(row,reason):
 return {**row,'status':'not_run','api_elapsed_ms':None,'elapsed_scope':'N/A','reason':reason}

def project_samples(planned,raw,step,parse_loss=False):
 """Retain actual sample events; missing evidence remains unknown, not invented timing."""
 keys={(r['trial_id'],r['sample_id']):i for i,r in enumerate(planned)}
 events={i:[] for i in range(len(planned))};errors=[];last=-1
 for row in raw:
  if type(row) is not dict or row.get('event') not in ['sample_start','sample_result','sample_terminal']:continue
  key=(row.get('trial_id'),row.get('sample_id'));i=keys.get(key)
  if i is None:errors.append('unknown sample identity');continue
  expected=planned[i]
  if row.get('query_id')!=expected['query_id'] or row.get('attempt')!=1 or type(row.get('attempt')) is not int:
   errors.append('query/attempt identity mismatch');continue
  if i<last:errors.append('nonmonotonic sample order')
  last=max(last,i);events[i].append(row)
 evidence_loss=parse_loss or not step.get('capture_integrity',True) or bool(errors) or step.get('stream_truncated',False) or not step.get('drain_complete',False)
 output=[]
 for i,expected in enumerate(planned):
  ev=events[i];starts=[r for r in ev if r['event']=='sample_start'];results=[r for r in ev if r['event']=='sample_result'];ends=[r for r in ev if r['event']=='sample_terminal']
  row={**expected,'status_scope':'adapter_terminal_outcome_only_not_validation','validation_status':'not_validated','api_elapsed_ms':None,'elapsed_scope':'N/A','start_monotonic_ns':None,'terminal_monotonic_ns':None}
  valid=(len(starts)==1 and len(results)<=1 and len(ends)<=1 and [r['event'] for r in ev] in [['sample_start'],['sample_start','sample_result'],['sample_start','sample_terminal'],['sample_start','sample_result','sample_terminal']])
  start=starts[0] if len(starts)==1 else None;end=ends[0] if len(ends)==1 else None
  if start:
   stamp=start.get('monotonic_ns')
   valid=valid and type(stamp) is int and stamp>=step.get('started_monotonic_ns',0) and start.get('clock')=='mach_absolute_time_ns' and type(start.get('numer')) is int and start['numer']>0 and type(start.get('denom')) is int and start['denom']>0
   row['start_monotonic_ns']=stamp
  if end and valid:
   elapsed=end.get('api_elapsed_ms');attempt=end.get('attempt_until_terminal_creation_ms');stamp=end.get('monotonic_ns')
   valid=(type(stamp) is int and stamp>=start['monotonic_ns'] and end.get('clock')=='mach_absolute_time_ns' and end.get('status') in ['success','error','panic'] and type(end.get('reason')) is str and len(end['reason'])<=512 and (elapsed is None or type(elapsed) in [int,float] and math.isfinite(elapsed) and elapsed>=0) and type(attempt) in [int,float] and math.isfinite(attempt) and attempt>=0 and abs(attempt-(stamp-start['monotonic_ns'])/1e6)<.00001)
   if end.get('status')=='success':valid=valid and len(results)==1 and elapsed is not None and end.get('api_elapsed_scope')=='completed_api_scope'
   if results:valid=valid and type(results[0].get('elapsed_ms')) in [float,int] and results[0]['elapsed_ms']==elapsed
   if valid:
    row.update(status=end['status'],reason=end['reason'],api_elapsed_ms=elapsed,elapsed_scope=end.get('api_elapsed_scope'),terminal_monotonic_ns=stamp,attempt_until_terminal_creation_ms=attempt)
    output.append(row);continue
  if ev and not valid:errors.append('invalid sample event sequence or timing: '+expected['sample_id'])
  # A complete API result can prove API elapsed even if terminal publication failed.
  if valid and len(results)==1:
   elapsed=results[0].get('elapsed_ms')
   if type(elapsed) in [int,float] and math.isfinite(elapsed) and elapsed>=0:row.update(api_elapsed_ms=elapsed,elapsed_scope='completed_api_result_without_terminal')
  if not ev and not evidence_loss and (not step.get('spawned') or i>last):
   row.update(status='not_run',elapsed_scope='N/A',reason='adapter serial order proves no start before process ended; '+str(step.get('stop_reason') or 'process_exit'))
  else:
   row.update(status='execution_unobserved',reason='no reliable terminal evidence; '+str(step.get('stop_reason') or 'process_exit'),duration_lower_bound_ms=0,lower_bound_scope='no positive query duration bound proven; zero is not successful elapsed time')
   stop=step.get('stop_observed_monotonic_ns')
   if start and type(stop) is int and type(start.get('monotonic_ns')) is int and stop>=start['monotonic_ns']:
    row['start_to_parent_stop_observation_window_ms']=(stop-start['monotonic_ns'])/1e6
  output.append(row)
 return output,errors

def read_events(path):
 rows=[];loss=False
 try:lines=path.read_text(errors='replace').splitlines()
 except BaseException as exc:return [{'event':'stream_read_error','error_type':type(exc).__name__,'error_message':str(exc)[:512]}],True
 for line in lines:
  if not line.strip():continue
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:loss=True
 return rows,loss

def comparisons(trials):
 output={'scope':'descriptive two historical full versions; no causal claim about scoring alone','quality':[],'scale':[]}
 for profile in PROFILES:
  for mode in ['text','vector','hybrid']:
   for cache in ['fresh_handle','warm_same_handle']:
    pair={}
    for role in POLICIES:
     selected=[t for t in trials if t['scope']=='quality' and not t['diagnostic'] and t['profile']==profile and t['policy']==role]
     rows=[r for t in selected for r in t['validated'] if r['mode']==mode and r['cache']==cache]
     need(len(rows)==96,'complete repeated quality group')
     pair[role]={'recall10':sum(r['recall10'] for r in rows)/96,'mrr10':sum(r['mrr10'] for r in rows)/96,'latency':summary([r['elapsed_ms'] for r in rows]),'rss_each_process':[t['rss_peak_bytes'] for t in selected]}
    output['quality'].append({'profile':profile,'mode':mode,'cache':cache,'versions':pair,'recall_delta':pair['complete_U']['recall10']-pair['window_scores']['recall10'],'mrr_delta':pair['complete_U']['mrr10']-pair['window_scores']['mrr10']})
  for mode in ['text','vector','hybrid','hybrid-filtered']:
   for scope in ['rust_search_execute','rust_search_standard_json']:
    pair={}
    for role in POLICIES:
     selected=[t for t in trials if t['scope']=='scale' and not t['diagnostic'] and t['profile']==profile and t['policy']==role]
     groups=[g for t in selected for g in t['validated'] if g['mode']==mode and g['scope']==scope]
     need(len(groups)==2,'complete scale paired rounds')
     pair[role]={'latency':summary([r['elapsed_ms'] for g in groups for r in g['all_samples'] if r['phase']=='measured']),'rss_each_process':[t['rss_peak_bytes'] for t in selected]}
    output['scale'].append({'profile':profile,'mode':mode,'scope':scope,'versions':pair,'p95_delta_ms':pair['complete_U']['latency']['p95_ms']-pair['window_scores']['latency']['p95_ms']})
 return output

def clock_domain():
 need(time.get_clock_info('monotonic').implementation=='mach_absolute_time()','Python monotonic clock implementation changed')
 before=time.monotonic_ns()
 row=json.loads(subprocess.check_output([str(ROOT/'sample-events-control'),'clock'],timeout=5,text=True))
 after=time.monotonic_ns()
 need(type(row.get('monotonic_ns')) is int and before<=row['monotonic_ns']<=after and type(row.get('numer')) is int and row['numer']>0 and type(row.get('denom')) is int and row['denom']>0,'parent/child clock origin/scale not proven')
 return {'before_ns':before,'child':row,'after_ns':after,'same_domain_proven':True}

def execute(protocol_path,output):
 protocol=json.loads(protocol_path.read_text())
 need(protocol.get('status')=='REPLAY_READY' and protocol.get('planned_trials')==plan(),'replay protocol/plan changed');identity(protocol)
 need(output.parent.resolve()==ROOT and re.fullmatch(r'run-[a-f0-9]{32}',output.name),'output must be a new owned nonce directory')
 clock=clock_domain()
 output.mkdir(mode=0o700)
 previous_signals={}
 def interrupted(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
 for number in [signal.SIGINT,signal.SIGTERM]:previous_signals[number]=signal.signal(number,interrupted)
 overall_start=time.monotonic();report={'status':'running','clock_domain':clock,'protocol_sha256':sha(protocol_path),'build_receipt_sha256':protocol.get('build_receipt_sha256'),'preparation':[],'trials':[],'not_run':plan(),'samples':[not_run_sample(row,'not yet dispatched') for i,item in enumerate(plan()) for row in sample_plan(item,f'trial-{i:02}')]};report_path=output/'report.json';save(report_path,report)
 try:
  for scope in ['quality','scale']:
   for role in POLICIES:
    db=output/(scope+'-'+role+'.coffer')
    manifests=[ROOT/'quality-input/manifest.json'] if scope=='quality' else [ROOT/f'scale-batches/input-{i}/manifest.json' for i in range(10)]
    for batch,manifest in enumerate(manifests):
     need(time.monotonic()-overall_start<2400,'overall preparation deadline');identity(protocol)
     argv=[str(ROOT/(role+'-target/release/coffer')),'import',str(db),'--manifest',str(manifest),'--json']
     step=run_process(argv,output/(scope+'-'+role+f'-import-{batch}'),min(300,max(.001,2400-(time.monotonic()-overall_start))))
     step.update(scope=scope,policy=role,batch=batch);report['preparation'].append(step);save(report_path,report);need(step['exit']==0 and step['stop_reason'] is None,'import failed')
     step['database_sha256']=sha(db)
  for i,item in enumerate(plan()):
   need(time.monotonic()-overall_start<=2400,'overall execution deadline');identity(protocol);role=item['policy'];scope=item['scope'];db=output/(scope+'-'+role+'.coffer');before=sha(db)
   binary=ROOT/(scope+'-'+role+('-reads' if item['diagnostic'] else ''))
   args=[str(binary),str(db),item['profile']]
   if scope=='quality':args += [str(ROOT/'quality-input/queries.jsonl'),str(ROOT/'quality-input/query_embeddings.jsonl')]
   trial_id=f'trial-{i:02}';args += [role,trial_id];base=output/trial_id
   trial={**item,'trial_id':trial_id,'status':'running'};report['trials'].append(trial);report['not_run'].pop(0);save(report_path,report)
   step=run_process(['/usr/bin/time','-l',*args],base,min(180,max(.001,2400-(time.monotonic()-overall_start))));trial.update(step)
   raw,loss=read_events(base.with_suffix('.stdout')) if step.get('stdout_sha256') is not None or 'stdout' in step.get('created_streams',[]) else ([],False)
   ledger,event_errors=project_samples(sample_plan(item,trial_id),raw,step,loss)
   report['samples']=[r for r in report['samples'] if r['trial_id']!=trial_id]+ledger
   trial['sample_event_errors']=event_errors;trial['parse_loss']=loss;trial['capture_read_errors']=[r for r in raw if type(r) is dict and r.get('event')=='stream_read_error']
   actual=sum(type(x) is dict and x.get('event')=='sample_result' for x in raw)
   trial['actual_query_records']=actual;trial['planned_query_records']=len(ledger)
   trial['unrecorded_planned_records']=max(0,len(ledger)-actual)
   if scope=='scale':
    trial['retained_result_validation_errors']=[]
    for row in raw:
     if type(row) is not dict or row.get('event')!='sample_result':continue
     try:row['candidate_layers']=validate_scale_result(row,role,item['profile'])
     except SampleValidationError as exc:trial['retained_result_validation_errors'].append({'sample_id':exc.sample_id,'reason':str(exc)})
    if trial['retained_result_validation_errors']:
     save(report_path,report);first=trial['retained_result_validation_errors'][0];raise SampleValidationError(first['sample_id'],first['reason'])
   save(report_path,report)
   need(not loss and not event_errors and all(r['status']=='success' for r in ledger),'sample event contract failed')
   need(step['exit']==0 and step['stop_reason'] is None,'trial failed');need(sha(db)==before,'readonly DB changed')
   trial['rss_peak_bytes']=rss(base.with_suffix('.stderr'));trial['database_sha256']=before
   rows=[r for r in raw if type(r) is not dict or r.get('event') not in ['sample_start','sample_terminal']];trial['validated']=validate_quality(rows,role,item['profile'],item['diagnostic']) if scope=='quality' else validate_scale(rows,role,item['profile'],item['diagnostic'])
   for row in report['samples']:
    if row['trial_id']==trial_id:row['validation_status']='passed'
   trial['status']='passed';save(report_path,report)
  identity(protocol);report['comparison']=comparisons(report['trials']);report['status']='passed'
 except BaseException as exc:
  report['status']='failed'
  if report['trials'] and report['trials'][-1]['status']=='running':report['trials'][-1]['status']='failed'
  report['error']=type(exc).__name__+': '+str(exc)
  if report['trials'] and report['trials'][-1]['status']=='failed':
   failed_id=report['trials'][-1].get('trial_id')
   for row in report['samples']:
    if row['trial_id']==failed_id:
     row['validation_status']='failed' if isinstance(exc,SampleValidationError) and row['sample_id']==exc.sample_id else 'not_completed_due_trial_failure'
     row['validation_reason']=report['error']
  for row in report['samples']:
   if row['status']=='not_run' and row['reason']=='not yet dispatched':row['reason']='upstream stopped: '+report['error']
  raise
 finally:
  save(report_path,report)
  for number,handler in previous_signals.items():signal.signal(number,handler)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--new-output',type=Path,required=True);a=p.parse_args();execute(a.protocol,a.new_output)
