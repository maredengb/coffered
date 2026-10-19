import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',P/'replay.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class Control(unittest.TestCase):
 def test_invalid_protocol_refuses_before_outputs_or_process(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'protocol.json').write_text('{}')
   with patch.object(s,'identity') as identity,patch.object(s,'run_process') as run:
    with self.assertRaises(ValueError):s.execute(p/'protocol.json',p/'no-output')
    identity.assert_not_called();run.assert_not_called();self.assertFalse((p/'no-output').exists())
 def test_process_failure_timeout_and_bounded_both_streams(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)
   cases=[('failure','import sys;print("partial");sys.stderr.write("err");sys.exit(3)',2,None,3),('timeout','import time;print("partial",flush=True);time.sleep(10)',.2,'timeout',None),('overflow','import os;os.write(1,b"x"*100000);os.write(2,b"y"*100000)',2,'log_limit',None)]
   for n,code,limit,reason,exit_ in cases:
    row=s.run_process([sys.executable,'-I','-B','-c',code],base/n,limit,max_output=1024)
    self.assertEqual(row['stop_reason'],reason)
    if exit_ is not None:self.assertEqual(row['exit'],exit_)
    self.assertLessEqual((base/(n+'.stdout')).stat().st_size,1024);self.assertLessEqual((base/(n+'.stderr')).stat().st_size,1024)
   self.assertIn(b'partial',(base/'timeout.stdout').read_bytes())
 def test_observed_storage_threshold_stops_owned_process(self):
  with tempfile.TemporaryDirectory() as d,patch.object(s,'storage_safe',return_value=False):
   row=s.run_process([sys.executable,'-I','-B','-c','import time;time.sleep(5)'],Path(d)/'storage',3)
   self.assertEqual(row['stop_reason'],'observed_storage_threshold');self.assertNotEqual(row['exit'],0)
 def test_closed_pipes_do_not_end_resource_monitor(self):
  with tempfile.TemporaryDirectory() as d,patch.object(s,'storage_safe',return_value=False) as guard:
   row=s.run_process([sys.executable,'-I','-B','-c','import os,time;os.close(1);os.close(2);time.sleep(5)'],Path(d)/'closed',3)
   self.assertEqual(row['stop_reason'],'observed_storage_threshold');self.assertNotEqual(row['exit'],0);guard.assert_called()
 def test_monitor_failure_preserves_structured_partial_receipt(self):
  with tempfile.TemporaryDirectory() as d,patch.object(s.subprocess,'check_output',side_effect=OSError('synthetic monitor failure')),patch.object(s,'storage_safe',return_value=True):
   base=Path(d)/'monitor'
   row=s.run_process([sys.executable,'-I','-B','-c','import time;print("partial",flush=True);time.sleep(5)'],base,3)
   self.assertEqual(row['stop_reason'],'process_exception');self.assertEqual(row['error_type'],'OSError');self.assertEqual(row['exit'],-9)
   self.assertEqual(row['stdout_sha256'],s.sha(base.with_suffix('.stdout')));self.assertIn('partial',base.with_suffix('.stdout').read_text())
 def test_spawn_failure_and_collision_do_not_claim_execution(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)/'spawn';row=s.run_process(['/nonexistent-cofferdb-control'],base,1)
   self.assertEqual(row['stop_reason'],'process_exception');self.assertEqual(row['error_type'],'FileNotFoundError');self.assertIsNone(row['exit'])
   row=s.run_process(['/nonexistent-cofferdb-control'],base,1)
   self.assertEqual(row['stop_reason'],'output_collision');self.assertIsNone(row['stdout_sha256'])
 def test_plan_exact_counts_and_reverse_order(self):
  plan=s.plan();self.assertEqual(len(plan),36)
  for scope in ['quality','scale']:
   primary=[r for r in plan if r['scope']==scope and not r['diagnostic']];diagnostic=[r for r in plan if r['scope']==scope and r['diagnostic']]
   self.assertEqual(len(primary),12);self.assertEqual(len(diagnostic),6)
   self.assertEqual([r['policy'] for r in primary[:2]],s.POLICIES);self.assertEqual([r['policy'] for r in primary[6:8]],list(reversed(s.POLICIES)))
  self.assertEqual(sum(288 for r in plan if r['scope']=='quality' and not r['diagnostic']),3456)
  self.assertEqual(sum(184 for r in plan if r['scope']=='scale' and not r['diagnostic']),2208)
 def test_read_scope_separates_instrumentation_from_primary(self):
  s.validate_reads(None,False)
  good={'scope':'querying_thread_userspace_read_pread_readv','calls':1,'known_requested_bytes':8,'returned_bytes':4,'errors':0,'saturated':False,'physical_read_bytes':None}
  s.validate_reads(good,True)
  for change in [{'physical_read_bytes':4},{'saturated':True},{'errors':1},{'returned_bytes':9},{'calls':True}]:
   with self.assertRaises(ValueError):s.validate_reads({**good,**change},True)
  with self.assertRaises(ValueError):s.validate_reads(good,False)
 def test_quality_metrics_use_actual_qrels_and_reject_unjudged(self):
  qrels=s.parse_rows(P/'quality-input/qrels.jsonl');positive={}
  for r in qrels:
   if r['relevance']>=1:positive.setdefault(r['query_id'],[]).append(r['chunk_external_id'])
  queries=[r['query_id'] for r in s.parse_rows(P/'quality-input/queries.jsonl')]
  rows=[{'mode':m,'query_id':q,'cache':c,'profile':'balanced','policy':'complete_U','elapsed_ms':1.0,'reads':None,'result':{'results':[{'chunk_external_id':positive[q][0]}]}} for m in ['text','vector','hybrid'] for q in queries for c in ['fresh_handle','warm_same_handle']]
  metrics=s.validate_quality(rows,'complete_U','balanced',False);self.assertEqual(len(metrics),288);self.assertTrue(all(r['mrr10']==1 for r in metrics))
  rows[0]['result']['results'][0]['chunk_external_id']='unjudged'
  with self.assertRaises(ValueError):s.validate_quality(rows,'complete_U','balanced',False)
 def test_nearest_rank_and_nonfinite_rejection(self):
  self.assertEqual(s.summary([4,1,2,3]),{'samples':4,'p50_ms':2,'p95_ms':4})
  for v in [[],[float('nan')],[-1]]:
   with self.assertRaises(ValueError):s.summary(v)
 def planned_control(self):
  item=s.plan()[0]
  return [dict(item,trial_id='trial-00',sample_id='control:sample',query_id='control-query',attempt=1),dict(item,trial_id='trial-00',sample_id='control:later',query_id='later-query',attempt=1)]
 def test_real_rust_error_and_panic_preserve_api_duration(self):
  for mode,code in [('error',1),('panic',101)]:
   with tempfile.TemporaryDirectory() as d:
    base=Path(d)/mode;step=s.run_process([str(P/'sample-events-control'),mode],base,2)
    raw,loss=s.read_events(base.with_suffix('.stdout'));rows,errors=s.project_samples(self.planned_control(),raw,step,loss)
    self.assertEqual(step['exit'],code);self.assertFalse(loss);self.assertEqual(errors,[])
    self.assertEqual(rows[0]['status'],mode);self.assertGreater(rows[0]['api_elapsed_ms'],0)
    self.assertEqual(rows[0]['elapsed_scope'],'api_until_error_observed_in_adapter')
    self.assertEqual(rows[1]['status'],'not_run');self.assertIsNone(rows[1]['api_elapsed_ms'])
 def test_timeout_start_is_unknown_not_positive_duration_lower_bound(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)/'timeout';code='import json,time;print(json.dumps(dict(event="sample_start",trial_id="trial-00",sample_id="control:sample",query_id="control-query",attempt=1,monotonic_ns=time.monotonic_ns(),clock="mach_absolute_time_ns",numer=125,denom=3)),flush=True);time.sleep(10)'
   step=s.run_process([sys.executable,'-I','-B','-c',code],base,.25)
   raw,loss=s.read_events(base.with_suffix('.stdout'));rows,errors=s.project_samples(self.planned_control(),raw,step,loss)
   self.assertEqual(step['stop_reason'],'timeout');self.assertLessEqual(step['stop_observed_monotonic_ns'],step['reaped_monotonic_ns'])
   self.assertEqual(errors,[]);self.assertEqual(rows[0]['status'],'execution_unobserved');self.assertIsNone(rows[0]['api_elapsed_ms']);self.assertEqual(rows[0]['duration_lower_bound_ms'],0)
   self.assertGreater(rows[0]['start_to_parent_stop_observation_window_ms'],0)
   self.assertEqual(rows[1]['status'],'not_run')
 def test_terminal_overrides_parent_stop_and_missing_terminal_is_unknown(self):
  plan=self.planned_control();base=dict(trial_id='trial-00',sample_id='control:sample',query_id='control-query',attempt=1)
  start=dict(base,event='sample_start',monotonic_ns=1000000,clock='mach_absolute_time_ns',numer=125,denom=3)
  result=dict(base,event='sample_result',elapsed_ms=.5)
  end=dict(base,event='sample_terminal',monotonic_ns=2000000,clock='mach_absolute_time_ns',status='success',reason='',api_elapsed_ms=.5,api_elapsed_scope='completed_api_scope',attempt_until_terminal_creation_ms=1)
  step=dict(started_monotonic_ns=0,stop_observed_monotonic_ns=1500000,drain_complete=True,spawned=True,stop_reason='timeout')
  rows,errors=s.project_samples(plan,[start,result,end],step);self.assertFalse(errors);self.assertEqual(rows[0]['status'],'success');self.assertEqual(rows[0]['api_elapsed_ms'],.5)
  rows,errors=s.project_samples(plan,[start,result],step);self.assertFalse(errors);self.assertEqual(rows[0]['status'],'execution_unobserved');self.assertEqual(rows[0]['api_elapsed_ms'],.5)
  rows,errors=s.project_samples(plan,[],dict(step,stream_truncated=True));self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
  rows,errors=s.project_samples(plan,[],step,True);self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
  rows,errors=s.project_samples(plan,[],dict(step,spawned=False));self.assertTrue(all(r['status']=='not_run' for r in rows))
 def test_all_planned_samples_have_unique_stable_identity(self):
  rows=[row for i,item in enumerate(s.plan()) for row in s.sample_plan(item,f'trial-{i:02}')]
  self.assertEqual(len(rows),7488);self.assertEqual(len({(r['trial_id'],r['sample_id']) for r in rows}),7488)
  self.assertTrue(all(r['attempt']==1 and r['query_id'] for r in rows))
  for row in rows:
   out=s.not_run_sample(row,'upstream stopped');self.assertIsNone(out['api_elapsed_ms']);self.assertEqual(out['elapsed_scope'],'N/A')
 def test_clock_domain_control_is_actual_and_no_engine(self):
  import time
  self.assertEqual(time.get_clock_info('monotonic').implementation,'mach_absolute_time()')
  before=time.monotonic_ns();row=json.loads(subprocess.check_output([str(P/'sample-events-control'),'clock']));after=time.monotonic_ns()
  self.assertLessEqual(before,row['monotonic_ns']);self.assertLessEqual(row['monotonic_ns'],after)
  self.assertGreater(row['numer'],0);self.assertGreater(row['denom'],0)
 def test_failed_first_trial_retains_full_ledger_and_future_NA(self):
  import shutil
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();(root/'quality-input').mkdir();shutil.copyfile(P/'quality-input/queries.jsonl',root/'quality-input/queries.jsonl')
   protocol=root/'protocol.json';protocol.write_text(json.dumps({'status':'REPLAY_READY','planned_trials':s.plan()}))
   output=root/('run-'+'0'*32)
   def fake(argv,out,timeout):
    if 'import' in argv:
     Path(argv[2]).write_bytes(b'SYNTHETIC NOT A DATABASE');return dict(exit=0,stop_reason=None)
    planned=s.sample_plan(s.plan()[0],'trial-00');row=planned[0]
    ident={k:row[k] for k in ['trial_id','sample_id','query_id','attempt']}
    events=[dict(ident,event='sample_start',monotonic_ns=1000000,clock='mach_absolute_time_ns',numer=125,denom=3),dict(ident,event='sample_terminal',monotonic_ns=2000000,clock='mach_absolute_time_ns',status='error',reason='synthetic query error',api_elapsed_ms=.5,api_elapsed_scope='api_until_error_observed_in_adapter',attempt_until_terminal_creation_ms=1)]
    out.with_suffix('.stdout').write_text(''.join(json.dumps(r)+'\n' for r in events));out.with_suffix('.stderr').write_text('synthetic control only')
    return dict(exit=1,stop_reason=None,spawned=True,drain_complete=True,started_monotonic_ns=0,stdout_sha256=s.sha(out.with_suffix('.stdout')))
   with patch.object(s,'ROOT',root),patch.object(s,'identity'),patch.object(s,'clock_domain',return_value={'synthetic':True}),patch.object(s,'run_process',side_effect=fake):
    with self.assertRaises(ValueError):s.execute(protocol,output)
   report=json.loads((output/'report.json').read_text());self.assertEqual(report['status'],'failed');self.assertEqual(len(report['samples']),7488);self.assertNotIn('comparison',report)
   first=[r for r in report['samples'] if r['trial_id']=='trial-00'];self.assertEqual(first[0]['status'],'error');self.assertEqual(first[0]['api_elapsed_ms'],.5)
   future=[r for r in report['samples'] if r['trial_id']!='trial-00'];self.assertTrue(all(r['status']=='not_run' and r['api_elapsed_ms'] is None and r['reason'].startswith('upstream stopped:') for r in future))
 def test_invalid_or_unknown_event_fails_without_fabricating_success(self):
  plan=self.planned_control();step=dict(started_monotonic_ns=0,spawned=True,drain_complete=True)
  rows,errors=s.project_samples(plan,[{'event':'sample_start','trial_id':'trial-99','sample_id':'unknown'}],step)
  self.assertTrue(errors);self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
  ident={k:plan[0][k] for k in ['trial_id','sample_id','query_id','attempt']}
  start=dict(ident,event='sample_start',monotonic_ns=1000000,clock='mach_absolute_time_ns',numer=125,denom=3)
  end=dict(ident,event='sample_terminal',monotonic_ns=2000000,clock='mach_absolute_time_ns',status='success',reason='',api_elapsed_ms=.5,api_elapsed_scope='completed_api_scope',attempt_until_terminal_creation_ms=1)
  rows,errors=s.project_samples(plan,[start,end],step);self.assertTrue(errors);self.assertEqual(rows[0]['status'],'execution_unobserved')
 def test_capture_hash_failure_preserves_receipt_and_unknown_states(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)/'hash'
   with patch.object(s,'sha',side_effect=OSError('synthetic hash read failure')):
    step=s.run_process([sys.executable,'-I','-B','-c','print("executed",flush=True)'],base,2)
   self.assertEqual(step['exit'],0);self.assertEqual(step['stop_reason'],'capture_finalize_exception');self.assertFalse(step['capture_integrity']);self.assertIsNone(step['stdout_sha256']);self.assertTrue(step['capture_errors']);self.assertIn('stdout',step['created_streams'])
   raw,loss=s.read_events(base.with_suffix('.stdout'));rows,errors=s.project_samples(self.planned_control(),raw,step,loss)
   self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
 def test_capture_read_failure_is_loss_and_never_not_run(self):
  with tempfile.TemporaryDirectory() as d:
   raw,loss=s.read_events(Path(d)/'missing.stdout');self.assertTrue(loss);self.assertEqual(raw[0]['event'],'stream_read_error')
   rows,errors=s.project_samples(self.planned_control(),raw,dict(spawned=True,drain_complete=True),loss)
   self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
 def test_quality_parent_failure_identifies_exact_sample(self):
  qrels=s.parse_rows(P/'quality-input/qrels.jsonl');positive={}
  for r in qrels:
   if r['relevance']>=1:positive.setdefault(r['query_id'],[]).append(r['chunk_external_id'])
  rows=[dict(r,profile='balanced',elapsed_ms=1.0,reads=None,result={'results':[{'chunk_external_id':positive[r['query_id']][0]}]}) for r in s.sample_plan(s.plan()[0],'trial-00')]
  for row in rows:row['policy']='complete_U'
  rows[7]['result']={'results':[{'chunk_external_id':'unjudged'}]}
  with self.assertRaises(s.SampleValidationError) as caught:s.validate_quality(rows,'complete_U','balanced',False)
  self.assertEqual(caught.exception.sample_id,rows[7]['sample_id'])
 def test_consumed_pipe_write_failure_and_short_write_are_capture_loss(self):
  original=Path.open
  for short in [False,True]:
   with tempfile.TemporaryDirectory() as d:
    base=Path(d)/'capture'
    class Broken:
     def __init__(self,wrapped):self.wrapped=wrapped
     def write(self,payload):
      if not payload:return 0
      if short:return self.wrapped.write(payload[:1])
      raise OSError('synthetic capture write failure after pipe read')
     def close(self):return self.wrapped.close()
    def opened(path,*args,**kwargs):
     result=original(path,*args,**kwargs)
     return Broken(result) if path==base.with_suffix('.stdout') and args and args[0]=='xb' else result
    with patch.object(Path,'open',opened):
     step=s.run_process([sys.executable,'-I','-B','-c','import time;print("executed",flush=True);time.sleep(2)'],base,1)
    self.assertEqual(step['stop_reason'],'process_exception');self.assertFalse(step['capture_integrity']);self.assertTrue(step['capture_errors'])
    raw,loss=s.read_events(base.with_suffix('.stdout'));rows,errors=s.project_samples(self.planned_control(),raw,step,loss)
    self.assertTrue(all(r['status']=='execution_unobserved' for r in rows))
 def scale_result_fixture(self,role='complete_U',profile='balanced',mode='hybrid'):
  cap={'quality-first':500,'balanced':200,'resource-first':50}[profile];b=0 if mode=='vector' else min(100,cap);v=0 if mode=='text' else cap;union=max(b,v);fusion=min(union,cap)
  scored=0 if mode=='text' else 5000 if mode=='hybrid-filtered' else 100000
  scan=100000 if mode=='hybrid-filtered' and role=='complete_U' else scored
  stats=dict(vector_scan_count=scan,distance_ops=scored*768,fusion_candidate_count=fusion);limits=dict(bm25=cap,vector=cap,fusion=cap)
  hits=[dict(rank=i+1,chunk_external_id=f'chunk-{76400+i}',final_score=1.,bm25_score=None if mode=='vector' else 2.,vector_score=None if mode=='text' else 1.,normalized_scores=dict(bm25=0 if mode=='vector' else 1,vector=None if mode=='text' else 1,metadata_boost=0),explain={'normalization':{'candidate_count':union if role=='complete_U' else fusion}}) for i in range(10)]
  result=dict(profile=profile.replace('-','_'),top_k=10,results=hits,candidate_counts=dict(bm25=b,vector=v,fusion=fusion) if role=='complete_U' else limits,execution_stats=stats,query_context=dict(candidate_limits=limits,index_source='persistent_checkpoint'))
  return dict(event='sample_result',sample_id='fixture:sample',mode=mode,returned_count=10,result_retained_before_validation=True,result=result,index_source='persistent_checkpoint',**stats)
 def test_scale_raw_oracle_and_layer_semantics_do_not_invent_legacy_U(self):
  for role in s.POLICIES:
   for profile in s.PROFILES:
    for mode in ['text','vector','hybrid','hybrid-filtered']:
     row=self.scale_result_fixture(role,profile,mode);layers=s.validate_scale_result(row,role,profile)
     self.assertEqual(layers['returned'],10)
     if role=='window_scores':
      self.assertIsNone(layers['complete_union']['count']);self.assertEqual(layers['complete_union']['status'],'not_measured');self.assertIsNone(layers['branch_recalled_after_filter']['bm25'])
     else:self.assertEqual(layers['complete_union']['status'],'measured');self.assertIsInstance(layers['branch_recalled_after_filter']['bm25'],int)
 def test_scale_wrong_ranks_scores_counts_and_missing_raw_are_rejected_and_retained(self):
  import copy
  for mutation in ['rank','score','id','norm','work','missing']:
   row=self.scale_result_fixture()
   if mutation=='rank':row['result']['results'][0]['rank']=9
   if mutation=='score':row['result']['results'][0]['final_score']=.5
   if mutation=='id':row['result']['results'][0]['chunk_external_id']='wrong-id'
   if mutation=='norm':row['result']['results'][0]['normalized_scores']['bm25']=0
   if mutation=='work':row['distance_ops']=1
   if mutation=='missing':del row['result']
   before=copy.deepcopy(row)
   with self.assertRaises(s.SampleValidationError) as caught:s.validate_scale_result(row,'complete_U','balanced')
   self.assertEqual(caught.exception.sample_id,'fixture:sample');self.assertEqual(row,before)
 def test_real_wrong_result_is_flushed_before_panic_and_api_time_retained(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)/'wrong';step=s.run_process([str(P/'sample-events-control'),'wrong_result'],base,2)
   raw,loss=s.read_events(base.with_suffix('.stdout'));ledger,errors=s.project_samples(self.planned_control(),raw,step,loss)
   self.assertEqual(step['exit'],101);self.assertFalse(loss);self.assertFalse(errors);self.assertEqual(ledger[0]['status'],'panic');self.assertEqual(ledger[0]['elapsed_scope'],'completed_api_scope');self.assertGreater(ledger[0]['api_elapsed_ms'],0)
   result=[r for r in raw if r.get('event')=='sample_result'][0]
   self.assertEqual(result['result']['results'][0],dict(rank=9,chunk_external_id='wrong-id',final_score=0));self.assertEqual(result['elapsed_ms'],ledger[0]['api_elapsed_ms'])
   with self.assertRaises(s.SampleValidationError):s.validate_scale_result(result,'complete_U','balanced')
 def test_scale_bad_retained_result_fails_whole_synthetic_run_before_statistics(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();item=dict(scope='scale',diagnostic=False,round=1,profile='balanced',policy='complete_U');trial='trial-00'
   protocol=root/'protocol.json';protocol.write_text(json.dumps({'status':'REPLAY_READY','planned_trials':[item]}))
   output=root/('run-'+'1'*32)
   planned=s.sample_plan(item,trial)[0];row=self.scale_result_fixture();row.update({k:planned[k] for k in ['sample_id','query_id','trial_id','attempt']});row.update(elapsed_ms=.5,stage='query');row['result']['results'][0]['rank']=9
   def fake(argv,out,timeout):
    if 'import' in argv:Path(argv[2]).write_bytes(b'SYNTHETIC NOT DB');return dict(exit=0,stop_reason=None)
    ident={k:planned[k] for k in ['sample_id','query_id','trial_id','attempt']}
    events=[dict(ident,event='sample_start',monotonic_ns=1000000,clock='mach_absolute_time_ns',numer=125,denom=3),row,dict(ident,event='sample_terminal',monotonic_ns=2000000,clock='mach_absolute_time_ns',status='panic',reason='synthetic assertion after retention',api_elapsed_ms=.5,api_elapsed_scope='completed_api_scope',attempt_until_terminal_creation_ms=1)]
    out.with_suffix('.stdout').write_text(''.join(json.dumps(r)+'\n' for r in events));out.with_suffix('.stderr').write_text('synthetic only')
    return dict(exit=101,stop_reason=None,spawned=True,drain_complete=True,started_monotonic_ns=0,stdout_sha256=s.sha(out.with_suffix('.stdout')))
   with patch.object(s,'ROOT',root),patch.object(s,'plan',return_value=[item]),patch.object(s,'identity'),patch.object(s,'clock_domain',return_value={'synthetic':True}),patch.object(s,'run_process',side_effect=fake):
    with self.assertRaises(s.SampleValidationError):s.execute(protocol,output)
   report=json.loads((output/'report.json').read_text());self.assertEqual(report['status'],'failed');self.assertNotIn('comparison',report)
   failed=[r for r in report['samples'] if r['sample_id']==planned['sample_id']][0];self.assertEqual(failed['validation_status'],'failed');self.assertEqual(failed['api_elapsed_ms'],.5)
   saved,loss=s.read_events(output/'trial-00.stdout');self.assertFalse(loss);self.assertEqual(saved[1]['result']['results'][0]['rank'],9);self.assertTrue(report['trials'][0]['retained_result_validation_errors'])
 def test_storage_scan_removed_temporary_file_is_absent_but_other_errors_stop(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stable=root/'stable';stable.write_bytes(b'abc');gone=root/'cofferdb-text-sort-control.tmp';gone.write_bytes(b'12345');(root/'ignored-link').symlink_to(stable)
   original=Path.lstat
   def removed(path,*args,**kwargs):
    if path==gone and gone.exists():gone.unlink()
    return original(path,*args,**kwargs)
   with patch.object(s,'ROOT',root),patch.object(Path,'lstat',removed):self.assertEqual(s.owned_logical_bytes(),3)
   def denied(path,*args,**kwargs):
    if path==stable:raise PermissionError('synthetic real scan failure')
    return original(path,*args,**kwargs)
   with patch.object(s,'ROOT',root),patch.object(Path,'lstat',denied):
    with self.assertRaises(PermissionError):s.owned_logical_bytes()
 def test_live_temp_churn_does_not_terminate_monitored_pure_child(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code='import pathlib,time; p=pathlib.Path("cofferdb-text-sort-control.tmp"); end=time.monotonic()+1.4\nwhile time.monotonic()<end:\n p.write_bytes(b"x"*4096);p.unlink()\nprint("pure churn complete",flush=True)'
   with patch.object(s,'ROOT',root):row=s.run_process([sys.executable,'-I','-B','-c',code],root/'churn',4)
   self.assertEqual(row['exit'],0);self.assertIsNone(row['stop_reason']);self.assertTrue(row['capture_integrity']);self.assertIn('pure churn complete',(root/'churn.stdout').read_text())
 def test_storage_scan_directory_enumeration_error_is_not_hidden(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with patch.object(s,'ROOT',root),patch.object(s.os,'scandir',side_effect=PermissionError('synthetic directory enumeration denied')):
    with self.assertRaises(PermissionError):s.owned_logical_bytes()
   gone=root/'gone-directory';gone.mkdir();(root/'stable').write_bytes(b'abc');original=s.os.scandir
   def disappeared(path):
    if Path(path)==gone:
     gone.rmdir();raise FileNotFoundError('synthetic directory removed before observation')
    return original(path)
   with patch.object(s,'ROOT',root),patch.object(s.os,'scandir',disappeared):self.assertEqual(s.owned_logical_bytes(),3)
if __name__=='__main__':unittest.main()
