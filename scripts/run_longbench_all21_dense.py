#!/usr/bin/env python3
"""Run LLaDA dense Q-Mode over all official LongBench tasks on one GPU."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT=Path('/data0/gs/SparseDLM_LLaDA_SDAR')
sys.path.insert(0,str(ROOT))
from src.runtime import load_runtime

TASKS='2wikimqa dureader gov_report hotpotqa lcc lsht multi_news multifieldqa_en multifieldqa_zh musique narrativeqa passage_count passage_retrieval_en passage_retrieval_zh qasper qmsum repobench-p samsum trec triviaqa vcsum'.split()

def args():
 p=argparse.ArgumentParser()
 p.add_argument('--data_dir',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True)
 p.add_argument('--model_path',required=True); p.add_argument('--limit',type=int,default=100000)
 p.add_argument('--max_context_tokens',type=int,default=32768); p.add_argument('--block_length',type=int,default=32)
 p.add_argument('--steps',type=int,default=32); p.add_argument('--threshold',type=float,default=0.7)
 p.add_argument('--editing_threshold',type=float,default=0.5); p.add_argument('--resume_from',type=Path,default=None)
 return p.parse_args()

def truncate_middle(ids,budget):
 n=int(ids.shape[-1])
 if n<=budget:return ids,False
 h=budget//2;t=budget-h
 return torch.cat((ids[:,:h],ids[:,-t:]),dim=-1),True

def main():
 a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((a.data_dir/'manifest.json').read_text())
 completed={}
 if a.resume_from and a.resume_from.exists():
  for line in a.resume_from.read_text(encoding='utf-8').splitlines():
   try:r=json.loads(line); completed[(r['task'],int(r['index']))]=r
   except: pass
 rows=list(completed.values()); skipped=0
 runtime=load_runtime('dense',family='llada',model_path=a.model_path,dtype='bfloat16',moe_expert_patch=True)
 model,tok=runtime.load()
 print(json.dumps({'moe_patched_blocks':runtime.moe_patch_report.patched_blocks,'tasks':len(manifest),'resumed':len(rows)}),flush=True)
 for item in manifest:
  task=item['task']; path=a.data_dir/f'{task}.jsonl'
  with path.open(encoding='utf-8') as f: records=[json.loads(x) for x in f if x.strip()][:a.limit]
  for idx,rec in enumerate(records):
   if (task,idx) in completed:
    skipped+=1;continue
   gen_length=int(item['max_gen'])
   ids=tok.apply_chat_template([{'role':'user','content':str(rec['prompt'])}],add_generation_prompt=True,tokenize=True,return_tensors='pt')
   original=int(ids.shape[-1]); budget=min(a.max_context_tokens-gen_length,int(tok.model_max_length)); budget=(budget//a.block_length)*a.block_length
   if budget<=0:raise ValueError('no prompt room')
   ids,trunc=truncate_middle(ids,budget); ids=ids.to(model.device)
   t0=time.perf_counter()
   with torch.inference_mode():
    out=runtime.generate(ids,gen_length=gen_length,block_length=a.block_length,steps=a.steps,temperature=0.0,
      threshold=a.threshold,mask_id=156895,eos_id=156892,eos_early_stop=True,editing_threshold=a.editing_threshold,num_to_transfer=1)
   row={'task':task,'index':idx,'original_input_tokens':original,'input_tokens':int(ids.shape[-1]),'truncated':trunc,
        'generated_tokens':int(out.tokens.shape[-1]),'elapsed_seconds':time.perf_counter()-t0,
        'prediction':tok.decode(out.tokens[0],skip_special_tokens=True),'answer':rec.get('answer'),
        'answers':rec.get('answers'),'all_classes':rec.get('all_classes'),'trace_events':len(out.trace)}
   rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
   report={'family':'llada','model':'LLaDA2.1','model_path':a.model_path,'mode':'dense','config':'Q Mode thr0.7/edit0.5',
     'block_length':a.block_length,'steps':a.steps,'moe_expert_patch':True,'moe_patched_blocks':runtime.moe_patch_report.patched_blocks,
     'data_dir':str(a.data_dir),'max_context_tokens':a.max_context_tokens,'rows':rows}
   (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'done_rows':len(rows),'skipped':skipped}),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
