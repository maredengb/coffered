// New supplement adapter; exact historical libraries, unchanged default weights.
use cofferdb::{parse_json, Database, DatabaseOpenOptions, QueryEmbeddingInput, ResourceProfile};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
include!("read_probe_bindings.rs");
include!("sample_events.rs");
fn string<'a>(v:&'a cofferdb::JsonValue,k:&str)->&'a str {v.get(k).unwrap().as_str().unwrap()}
fn main()->Result<(),Box<dyn Error>> {
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),7);
 let dbpath=Path::new(&args[1]);let profile=ResourceProfile::parse(&args[2]).expect("frozen profile");
 let policy=&args[5];let trial=&args[6];assert!(matches!(policy.as_str(),"window_scores"|"complete_U"));
 let queries:Vec<_>=fs::read_to_string(&args[3])?.lines().map(|x|parse_json(x).unwrap()).collect();
 assert_eq!(queries.len(),48);
 let mut vectors=BTreeMap::new();
 for line in fs::read_to_string(&args[4])?.lines() {let value=parse_json(line)?;let id=string(&value,"query_id").to_owned();assert!(vectors.insert(id,value).is_none());}
 assert_eq!(vectors.len(),48);
 for mode in ["text","vector","hybrid"] {for item in &queries {
  let id=string(item,"query_id");let vector=&vectors[id];
  let db=Database::open_with_options(dbpath,DatabaseOpenOptions{read_only:true,resource_profile:Some(profile),..DatabaseOpenOptions::default()})?;
  let collection=db.collection("supplement-quality")?;
  let mut query=collection.search().top_k(10).resource_profile(profile).filter(item.get("filter").unwrap().clone());
  assert_eq!(item.get("top_k").unwrap().as_f64(),Some(10.0));
  if mode!="vector" {query=query.text(string(item,"text"));}
  if mode!="text" {query=query.query_embedding(QueryEmbeddingInput{model_name:Some(string(vector,"model_name").into()),dim:384,vector:vector.get("vector").unwrap().as_array().unwrap().iter().map(|x|x.as_f64().unwrap() as f32).collect()});}
  for cache in ["fresh_handle","warm_same_handle"] {
   let call=query.clone();let sample_id=format!("quality:{mode}:{id}:{cache}");
   with_sample(trial,&sample_id,id,|sample| {
    sample.begin_api();let result=call.execute()?;let (elapsed,reads)=sample.finish_api();
    let wire=result.to_json().to_compact_string();
    sample_output(&format!("{{\"event\":\"sample_result\",\"trial_id\":{},\"sample_id\":{},\"attempt\":1,\"query_id\":{},\"mode\":\"{mode}\",\"profile\":\"{}\",\"policy\":\"{policy}\",\"cache\":\"{cache}\",\"elapsed_ms\":{elapsed},\"reads\":{reads},\"result\":{wire}}}",sample_json_string(trial),sample_json_string(&sample_id),sample_json_string(id),profile.as_json_str()))?;
    Ok(())
   })?;
  }
 }}
 Ok(())
}
