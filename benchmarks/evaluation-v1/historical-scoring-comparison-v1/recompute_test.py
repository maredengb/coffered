"""Pure data-integrity controls for the archived metric recomputation tool."""
from pathlib import Path
import gzip,hashlib,json,tempfile,unittest
import recompute

def sha(data):return hashlib.sha256(data).hexdigest()
class MetricControls(unittest.TestCase):
    def test_nearest_rank_uses_unrounded_pooled_samples(self):
        self.assertEqual(recompute.quantiles(list(reversed(range(1,41)))),{'samples':40,'p50_ms':20,'p95_ms':38})
        with self.assertRaises(ValueError):recompute.equal_numeric({'p95':38},{'p95':39})
    def test_invalid_or_negative_times_refused(self):
        for values in [[],[-1],[float('nan')],[float('inf')],[True]]:
            with self.assertRaises(ValueError):recompute.quantiles(values)
    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'result.json';out.write_bytes(b'keep existing bytes')
            with self.assertRaisesRegex(ValueError,'already exists'):recompute.recompute(root,root/'absent-qrels',out)
            self.assertEqual(out.read_bytes(),b'keep existing bytes')
    def test_duplicate_judgments_refused_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'trial-index.json').write_text('{}');q=root/'qrels';line=json.dumps({'query_id':'synthetic','chunk_external_id':'chunk','relevance':1})+'\n';q.write_text(line+line)
            with self.assertRaisesRegex(ValueError,'duplicate qrel'):recompute.recompute(root,q,root/'output')
            self.assertFalse((root/'output').exists())
    def fixture(self, root, path='data'):
        q=root/'qrels';rows=[]
        for query in range(48):
            for chunk in range(83 if query<24 else 82):rows.append({'query_id':f'q{query}','chunk_external_id':f'c{chunk}','relevance':1 if chunk==0 else 0})
        q.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.assertEqual(len(rows),3960)
        (root/'trial-index.json').write_text(json.dumps({'quality_qrels_sha256':recompute.digest(q)}))
        data=b'synthetic bytes';(root/'data').write_bytes(data)
        row={'path':path,'bytes':len(data),'sha256':sha(data)}
        return q,row
    def test_changed_files_and_escape_paths_refused(self):
        for mode in ['hash','escape']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);q,row=self.fixture(root,path='../outside' if mode=='escape' else 'data')
                if mode=='hash':(root/'data').write_bytes(b'changed bytes')
                (root/'lineage.json').write_text(json.dumps({'files':[row]}))
                with self.assertRaises(ValueError):recompute.recompute(root,q,root/'output')
                self.assertFalse((root/'output').exists())
    def test_directory_symlink_is_not_followed_even_with_matching_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);q,row=self.fixture(root);external=root/'outside';external.mkdir();(external/'data').write_bytes(b'synthetic bytes');(root/'linked').symlink_to(external,target_is_directory=True);row['path']='linked/data'
            (root/'lineage.json').write_text(json.dumps({'files':[row]}))
            with self.assertRaisesRegex(ValueError,'changed file'):recompute.recompute(root,q,root/'output')
            self.assertFalse((root/'output').exists())
    def test_gzip_expanded_digest_and_size_bounds_checked(self):
        for mode in ['digest','size']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);q,row=self.fixture(root);payload=b'synthetic expanded bytes';data=gzip.compress(payload);(root/'data.gz').write_bytes(data)
                row.update(path='data.gz',bytes=len(data),sha256=sha(data),storage='gzip',uncompressed_bytes=1 if mode=='size' else len(payload),uncompressed_sha256='wrong')
                (root/'lineage.json').write_text(json.dumps({'files':[row]}))
                with self.assertRaisesRegex(ValueError,'expanded file exceeds bound|compressed payload changed'):recompute.recompute(root,q,root/'output')
                self.assertFalse((root/'output').exists())

if __name__=='__main__':unittest.main()
