"""Pure filesystem/argument controls; no database or compiler execution."""
import hashlib, io, json, os, tempfile, unittest, tarfile
from pathlib import Path
from unittest.mock import patch
import prepare, build

def sha(data):return hashlib.sha256(data).hexdigest()
def row(path, data):return {'path':path,'bytes':len(data),'sha256':sha(data)}

class PreparationControls(unittest.TestCase):
    def fixture(self, root):
        package=root/'package';evaluation=root/'evaluation';package.mkdir();evaluation.mkdir()
        (package/'sources/old/src').mkdir(parents=True);(package/'sources/old/src/lib.rs').write_bytes(b'// synthetic fixture\n')
        source=[row('sources/old/src/lib.rs',b'// synthetic fixture\n')]
        (package/'sources/README.md').write_bytes(b'Synthetic source readme.\n')
        manifest=json.dumps({'files':source}).encode();(package/'source-files.json').write_bytes(manifest)
        quality=evaluation/'public-quality-v1';quality.mkdir();q=[]
        for name in ['documents.jsonl','chunks.jsonl','embeddings.jsonl','queries.jsonl','query_embeddings.jsonl','qrels.jsonl']:
            data=b'{}\n';(quality/name).write_bytes(data);q.append({'name':name,'bytes':len(data),'sha256':sha(data)})
        assets=package/'assets';assets.mkdir();payload=b'{}\n'
        wrapper={'collection':'synthetic','documents':'documents.jsonl','chunks':'chunks.jsonl','embeddings':'embeddings.jsonl','hashes':{k+'_sha256':sha(payload) for k in ['documents','chunks','embeddings']}}
        for name in ['quality-manifest.json',*[f'scale-manifest-{i}.json' for i in range(10)]]:(assets/name).write_text(json.dumps(wrapper))
        for name in ['control.rs','probe.c']:(assets/name).write_bytes(b'// synthetic control fixture\n')
        for name in ['replay.py','replay_test.py','build.py','derivation.json']:(package/name).write_text('# synthetic control fixture\n')
        archive=evaluation/'logical.tar.gz';members=[]
        with tarfile.open(archive,'w:gz') as tar:
            names=[f'input/input-{i}/{n}.jsonl' for i in range(10) for n in ['documents','chunks','embeddings']]+[f'extra-{i}' for i in range(14)]
            for name in names:
                info=tarfile.TarInfo(name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload));members.append({'path':name,'bytes':len(payload),'public_member_sha256':sha(payload)})
        spec={'source_manifest_sha256':sha(manifest),'source_readme':row('sources/README.md',b'Synthetic source readme.\n'),'assets':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())} for f in assets.iterdir()],'quality_files':q,'scale_archive':{'relative_path':'logical.tar.gz','sha256':sha(archive.read_bytes()),'members':members}}
        (package/'input-spec.json').write_text(json.dumps(spec))
        return package,evaluation
    def call(self, package, evaluation, output):
        with patch.object(prepare,'PACKAGE',package),patch.object(prepare.os,'statvfs',return_value=type('Space',(),{'f_bavail':40*1024**3,'f_frsize':1})()):
            prepare.prepare(evaluation,output)
    def test_source_and_input_subdirectories_refused_before_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package,evaluation=self.fixture(root)
            for output in [package/'sources/nested',package/'new',evaluation/'new']:
                with self.assertRaisesRegex(ValueError,'overlaps'):self.call(package,evaluation,output)
                self.assertFalse(output.exists())
    def test_unknown_source_directory_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package,evaluation=self.fixture(root);external=root/'external';external.mkdir();(external/'marker').write_bytes(b'synthetic external marker')
            (package/'sources/extra').symlink_to(external,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'links/special'):self.call(package,evaluation,root/'work')
            self.assertFalse((root/'work').exists())
    def test_source_file_and_assets_links_refused(self):
        for kind in ['source','asset','assets-directory']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);package,evaluation=self.fixture(root);marker=root/'marker';marker.write_bytes(b'synthetic marker')
                if kind=='source':path=package/'sources/old/src/lib.rs';path.unlink();path.symlink_to(marker)
                elif kind=='asset':path=package/'assets/probe.c';path.unlink();path.symlink_to(marker)
                else:(package/'assets').rename(package/'saved-assets');(package/'assets').symlink_to(package/'saved-assets',target_is_directory=True)
                with self.assertRaises(ValueError):self.call(package,evaluation,root/'work')
                self.assertFalse((root/'work').exists())
    def test_changed_readme_asset_missing_and_extra_refused_before_creation(self):
        for change in ['readme','asset','missing','extra']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);package,evaluation=self.fixture(root)
                if change=='readme':(package/'sources/README.md').write_bytes(b'changed')
                elif change=='asset':(package/'assets/probe.c').write_bytes(b'changed')
                elif change=='missing':(package/'assets/probe.c').unlink()
                else:(package/'assets/extra.c').write_bytes(b'extra')
                with self.assertRaises(ValueError):self.call(package,evaluation,root/'work')
                self.assertFalse((root/'work').exists())
    def test_exact_preparation_no_external_marker_and_existing_destination_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package,evaluation=self.fixture(root);out=root/'work';self.call(package,evaluation,out)
            self.assertEqual(len(list((out/'scale-batches').glob('*/embeddings.jsonl'))),10)
            self.assertEqual((out/'sources/old/src/lib.rs').read_bytes(),b'// synthetic fixture\n')
            self.assertFalse((out/'extra-0').exists())
            with self.assertRaisesRegex(ValueError,'already exists'):self.call(package,evaluation,out)
    def test_changed_quality_or_archive_refused(self):
        for change in ['quality','archive']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);package,evaluation=self.fixture(root)
                path=evaluation/'public-quality-v1/queries.jsonl' if change=='quality' else evaluation/'logical.tar.gz';path.write_bytes(b'changed')
                with self.assertRaises(ValueError):self.call(package,evaluation,root/'work')
                self.assertFalse((root/'work').exists())
    def test_safe_relative_rejects_absolute_parent_and_noncanonical(self):
        for path in ['/tmp/out','../out','a/../../out','a//b','a/./b']:
            with self.assertRaises(ValueError):prepare.safe_relative(path)

class CompilerControls(unittest.TestCase):
    def test_prepared_source_changed_during_build_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.rs';p.write_bytes(b'original');spec={'files':[{'path':'source.rs','bytes':8,'sha256':sha(b'original')}]}
            with patch.object(build,'ROOT',root):
                build.verify_preparation(spec)
                p.write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError,'prepared file changed'):build.verify_preparation(spec)
    def test_environment_drops_inherited_compiler_and_disables_flags_wrappers(self):
        env=build.build_environment({'RUSTC':'wrong','RUSTC_WRAPPER':'wrong','RUSTC_WORKSPACE_WRAPPER':'wrong','RUSTFLAGS':'wrong','CARGO_ENCODED_RUSTFLAGS':'wrong','CARGO_BUILD_TARGET':'wrong','CARGO_TARGET_DIR':'wrong','PATH':'test'})
        self.assertNotIn('RUSTC',env);self.assertNotIn('CARGO_BUILD_TARGET',env);self.assertNotIn('CARGO_TARGET_DIR',env)
        for key in ['RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS']:self.assertEqual(env[key],'')
        self.assertEqual(env['COFFERDB_GIT_COMMIT'],'unknown')
    def test_proxy_and_distinct_path_rustup_resolve_via_selected_sysroot(self):
        for hardlink in [False, True]:
            with self.subTest(hardlink=hardlink), tempfile.TemporaryDirectory() as d:
                root=Path(d);shims=root/'shims';shims.mkdir();mux=shims/'rustup';mux.write_bytes(b'synthetic mux')
                selector=shims/'rustc'
                if hardlink:os.link(mux,selector)
                else:selector.symlink_to(mux)
                other=root/'other-rustup';other.write_bytes(b'different synthetic mux')
                native=root/'native';(native/'bin').mkdir(parents=True);actual=native/'bin/rustc';actual.write_bytes(b'synthetic actual compiler')
                calls=[];env={'RUSTC':'wrong'}
                def run(name,argv):calls.append((name,argv));return str(native)+'\n'
                with patch.object(build.shutil,'which',side_effect=lambda n:str(selector) if n=='rustc' else str(other)):
                    chosen=build.select_compiler(env,run)
                self.assertEqual(chosen,str(actual.resolve()));self.assertEqual(env['RUSTC'],chosen)
                self.assertEqual(calls,[('compiler-sysroot',[str(selector),'--print','sysroot'])])
    def test_direct_native_compiler_needs_no_rustup(self):
        with tempfile.TemporaryDirectory() as d:
            native=Path(d).resolve();(native/'bin').mkdir();compiler=native/'bin/rustc';compiler.write_bytes(b'synthetic native compiler');env={}
            with patch.object(build.shutil,'which',side_effect=lambda n:str(compiler) if n=='rustc' else None):
                chosen=build.select_compiler(env,lambda name,argv:str(native))
            self.assertEqual(chosen,str(compiler));self.assertEqual(env['RUSTC'],chosen)
    def test_sysroot_pointing_back_to_proxy_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            native=Path(d).resolve();(native/'bin').mkdir();proxy=native/'bin/rustup';proxy.write_bytes(b'synthetic proxy');os.link(proxy,native/'bin/rustc')
            with patch.object(build.shutil,'which',side_effect=lambda n:str(native/'bin'/n)):
                with self.assertRaisesRegex(ValueError,'proxy'):build.select_compiler({},lambda *args:str(native))

if __name__=='__main__':unittest.main()
