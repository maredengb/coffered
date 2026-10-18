"""Build historical replay tools; separate pure controls from database execution."""
from pathlib import Path
import argparse, hashlib, json, os, platform, shutil, signal, subprocess, sys, time
ROOT = Path(__file__).resolve().parent
TARGET = 'aarch64-apple-darwin'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check(value, reason):
    if not value: raise ValueError(reason)

def build_environment(base):
    env = dict(base)
    for name in ['CARGO_BUILD_TARGET','CARGO_TARGET_DIR','RUSTC']:
        env.pop(name, None)
    for name in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER']:
        env[name] = ''
    env['COFFERDB_GIT_COMMIT'] = 'unknown'
    return env

def select_compiler(env, run):
    selector = shutil.which('rustc'); check(selector is not None, 'rustc is unavailable')
    sysroot = run('compiler-sysroot', [selector, '--print', 'sysroot']).strip()
    check(Path(sysroot).is_absolute(), 'compiler sysroot must be absolute')
    compiler = (Path(sysroot) / 'bin/rustc').resolve(strict=True)
    check(compiler.is_file(), 'native compiler is absent from the selected sysroot')
    # Rustup uses both symlinks and hard links; PATH may contain another rustup install.
    candidates = [Path(selector).with_name('rustup')]
    path_rustup = shutil.which('rustup')
    if path_rustup:candidates.append(Path(path_rustup))
    for candidate in candidates:
        if candidate.is_file():
            check(not compiler.samefile(candidate) and digest(compiler) != digest(candidate), 'sysroot compiler resolves to a toolchain proxy')
    env['RUSTC'] = str(compiler)
    return str(compiler)

def verify_preparation(preparation):
    for row in preparation['files']:
        path = ROOT / row['path']
        check(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'prepared file changed: ' + row['path'])

def build(controls_only):
    check(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'replay tooling requires macOS arm64')
    preparation = json.loads((ROOT / 'preparation.json').read_text())
    verify_preparation(preparation)
    stage = 'controls' if controls_only else 'engines'
    log = ROOT / ('build-' + stage); log.mkdir(mode=0o700)
    receipt = {'schema':1, 'stage':stage, 'status':'running', 'steps':[], 'platform':{'system':platform.system(), 'machine':platform.machine(), 'kernel':platform.release()}, 'engine_queries_executed':False}
    env = build_environment(os.environ)
    def run(name, argv, timeout=120):
        index = len(receipt['steps']); before = time.monotonic()
        stdout = log / f'{index:02}-{name}.stdout'; stderr = log / f'{index:02}-{name}.stderr'
        step = {'name':name, 'argv':list(map(str,argv)), 'exit':None, 'wall_ms':None}
        receipt['steps'].append(step)
        try:
            with stdout.open('xb') as out, stderr.open('xb') as err:
                process = subprocess.Popen(list(map(str,argv)), stdout=out, stderr=err, env=env, cwd=ROOT, start_new_session=True)
                try:
                    process.wait(timeout=timeout)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            step['exit'] = process.returncode
        finally:
            step['wall_ms'] = (time.monotonic() - before) * 1000
            step['stdout_sha256'] = digest(stdout) if stdout.exists() else None
            step['stderr_sha256'] = digest(stderr) if stderr.exists() else None
            (log/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        check(step['exit'] == 0, 'build/control failed: ' + name)
        return stdout.read_text()
    try:
        rustc = select_compiler(env, run)
        receipt['rustc_executable'] = rustc
        receipt['rustc_executable_sha256'] = digest(Path(rustc))
        receipt['rustc'] = run('rustc-version',[rustc,'-vV'])
        receipt['cargo'] = run('cargo-version',['cargo','-V'])
        receipt['clang'] = run('clang-version',['/usr/bin/clang','--version'])
        check('host: '+TARGET in receipt['rustc'], 'native Rust target mismatch')
        if controls_only:
            sdk = run('sdk',['/usr/bin/xcrun','--show-sdk-path']).strip()
            resource = run('clang-resource',['/usr/bin/clang','-print-resource-dir']).strip()
            cflags = ['/usr/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-nostdinc','-isystem',str(Path(sdk)/'usr/include'),'-isystem',str(Path(resource)/'include'),'-isysroot',sdk]
            dylib = ROOT/'libsupplement_read_probe.dylib'
            run('probe-library',[*cflags,'-dynamiclib',ROOT/'read_probe.c','-o',dylib,'-Wl,-install_name,'+str(dylib)])
            run('probe-c-build',[*cflags,ROOT/'read_probe_control.c','-L'+str(ROOT),'-lsupplement_read_probe','-o',ROOT/'read-probe-control'])
            run('probe-c-control',[ROOT/'read-probe-control'])
            run('probe-rust-build',[rustc,'--edition=2021','--target',TARGET,ROOT/'read_probe_rust_control.rs','-L',ROOT,'-l','dylib=supplement_read_probe','-o',ROOT/'read-probe-rust-control'])
            run('probe-rust-control',[ROOT/'read-probe-rust-control',ROOT/'probe-control-data'])
            run('sample-control-build',[rustc,'--edition=2021','--target',TARGET,ROOT/'sample_events_control.rs','-o',ROOT/'sample-events-control'])
            run('runner-controls',[sys.executable,'-B','-m','unittest','-v','replay_test.py'])
        else:
            control = json.loads((ROOT/'build-controls/receipt.json').read_text())
            check(control['status']=='passed', 'pure controls must pass first')
            for row in control['outputs']:
                check(digest(ROOT/row['path'])==row['sha256'], 'control binary changed')
            check(receipt['rustc']==control['rustc'] and receipt['rustc_executable_sha256']==control['rustc_executable_sha256'] and receipt['cargo']==control['cargo'] and receipt['clang']==control['clang'], 'compiler changed after controls')
            for role in ['window_scores','complete_U']:
                target = ROOT/('cargo-target-'+role)
                run('cargo-'+role,['cargo','build','--locked','--offline','--release','--target',TARGET,'--target-dir',target,'--manifest-path',ROOT/'sources'/role/'Cargo.toml','--lib','--bin','coffer'],timeout=300)
                release = target/TARGET/'release'
                cli = ROOT/(role+'-target/release/coffer');cli.parent.mkdir(parents=True);shutil.copyfile(release/'coffer',cli);cli.chmod(0o700)
                for scope in ['quality','scale']:
                    for probe in [False,True]:
                        name = scope+'-'+role+('-reads' if probe else '')
                        argv=[rustc,'--edition=2021','--target',TARGET,'-O',ROOT/(scope+'_helper.rs'),'--extern','cofferdb='+str(release/'libcofferdb.rlib'),'-L','dependency='+str(release/'deps'),'-o',ROOT/name]
                        if probe: argv += ['--cfg','read_probe','-L',str(ROOT),'-l','dylib=supplement_read_probe']
                        run(name,argv)
        verify_preparation(preparation)
        check(digest(Path(rustc)) == receipt['rustc_executable_sha256'], 'compiler changed during build')
        receipt['status']='passed'
    except BaseException as error:
        receipt['status']='failed';receipt['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        outputs = ['libsupplement_read_probe.dylib','read-probe-control','read-probe-rust-control','sample-events-control']
        if not controls_only: outputs += [role+'-target/release/coffer' for role in ['window_scores','complete_U']] + [scope+'-'+role+suffix for role in ['window_scores','complete_U'] for scope in ['quality','scale'] for suffix in ['','-reads']]
        receipt['outputs']=[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':digest(ROOT/name)} for name in outputs if (ROOT/name).is_file()]
        (log/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not controls_only:
        import replay
        files = []
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if any(part.startswith('cargo-target-') or part=='__pycache__' or part.startswith('run-') for part in relative.parts):continue
            check(not path.is_symlink(), 'prepared symlink')
            if path.is_file():files.append({'path':str(relative),'bytes':path.stat().st_size,'sha256':digest(path)})
        protocol={'schema':1,'status':'REPLAY_READY','preparation_root':str(ROOT),'files':files,'planned_trials':replay.plan(),'build_receipt_sha256':digest(log/'receipt.json'),'scope':'New historical replay identity, not an amendment to archived observations.','resource_observation':{'import_timeout_s':300,'trial_timeout_s':180,'overall_timeout_s':2400,'per_stream_limit_bytes':64*1024**2,'sampled_owned_group_rss_limit_bytes':4*1024**3,'sampled_owned_logical_limit_bytes':8*1024**3,'sampled_free_reserve_bytes':20*1024**3},'platform':receipt['platform'],'rustc':receipt['rustc'],'cargo':receipt['cargo'],'clang':receipt['clang'],'rustc_executable_sha256':receipt['rustc_executable_sha256']}
        with (ROOT/'protocol.json').open('x') as out:json.dump(protocol,out,indent=2);out.write('\n')
    print(json.dumps({'stage':stage,'status':receipt['status'],'receipt':str(log/'receipt.json'),'engine_queries_executed':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args();build(args.controls_only)
