"""One-shot dependency queue. CPU only; never stops/reserves any GPU."""
import fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
QUEUE=ROOT/'results/narrative80_focus_gpu4_20260917/queue'

def identity(pid):
    try:
        tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':tail[0],'start_ticks':tail[19]}
    except (FileNotFoundError,ProcessLookupError):return None

def write_state(phase,**kw):
    state=dict(phase=phase,queue_pid=os.getpid(),updated=datetime.now(timezone.utc).isoformat(),**kw)
    tmp=QUEUE/'state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(QUEUE/'state.json')
    print(json.dumps(state),flush=True)

def main():
    lock=(QUEUE/'lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((QUEUE/'dependency.json').read_text())
    formal=ROOT/'results/narrative80_focus_gpu4_20260917/formal'
    if formal.exists() or (QUEUE/'launch_once').exists():
        raise RuntimeError('Formal output or launch marker already exists; refusing duplicate')
    dep=cfg['pid'];ticks=cfg['start_ticks']
    write_state('waiting_for_losa_launcher',dependency=cfg,poll_seconds=2)
    while True:
        current=identity(dep)
        if not current or current['start_ticks']!=ticks or current['state'] in ('Z','X'):break
        time.sleep(2)
    write_state('waiting_for_gpu4_idle',dependency=cfg)
    while True:
        try:
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
            busy=any(line.split(',')[0].strip()==cfg['gpu_uuid'] for line in apps.splitlines())
            used=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
            if not busy and used<=256:break
        except (subprocess.SubprocessError,ValueError,OSError) as e:
            print('GPU status temporarily unavailable:',str(e),flush=True)
        time.sleep(2)
    with (QUEUE/'launch_once').open('x') as f:f.write(datetime.now(timezone.utc).isoformat())
    command=['bash','scripts/original/performance/run_narrative80_focus_gpu4.sh','--run']
    with (QUEUE.parent/'launcher.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write_state('focus_running',dependency=cfg,launcher_pid=child.pid,launcher_identity=identity(child.pid),command=command)
        rc=child.wait()
    cells=[formal/f'{m}_{n}_focus_optimized'/'COMPLETE.json' for m in ('sdar','llada21') for n in (8192,16384,32768)]
    complete=rc==0 and all(p.exists() for p in cells) and (formal/'summary.json').exists()
    write_state('completed' if complete else 'failed',dependency=cfg,exit_code=rc,complete_cells=sum(p.exists() for p in cells),summary_exists=(formal/'summary.json').exists(),gpu4_reserved_afterwards=False)
    return 0 if complete else 1

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        write_state('queue_error',error=repr(e))
        raise
