"""Opt-in per-model RoPE dispatch; original module globals stay untouched."""
import ast
import inspect
import textwrap
import types
import torch
from .rope_exact import apply

def eligible(q,k,c,s):
    return (not torch.is_grad_enabled() and q.is_cuda and 0<q.shape[-2]<=32
            and q.dtype in (torch.float16,torch.bfloat16,torch.float32) and q.shape[-1]==128 and q.dtype==k.dtype==c.dtype==s.dtype
            and c.shape==(q.shape[0],q.shape[-2],q.shape[-1]))

def _clone(function):
    function=getattr(function,'__func__',function)
    ns={**function.__globals__}
    if 'apply_rotary_pos_emb' in ns:
        reference=ns['apply_rotary_pos_emb']
        def pair(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
            if unsqueeze_dim==1 and eligible(q,k,cos,sin):return apply(q,k,cos,sin)
            return reference(q,k,cos,sin,position_ids=position_ids,unsqueeze_dim=unsqueeze_dim)
        ns['apply_rotary_pos_emb']=pair
        clone=types.FunctionType(function.__code__,ns,function.__name__,function.__defaults__,function.__closure__)
        clone.__kwdefaults__=function.__kwdefaults__
        return clone
    if '_apply_rotary' not in ns:return None
    single=ns['_apply_rotary']
    def pair(q,k,cos,sin):
        if eligible(q,k,cos,sin):return apply(q,k,cos,sin)
        return single(q,cos,sin),single(k,cos,sin)
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    fn=tree.body[0];fn.decorator_list=[]
    replacements=0
    for i in range(len(fn.body)-1):
        a,b=fn.body[i:i+2]
        def match(n):
            return (isinstance(n,ast.Assign) and len(n.targets)==1
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                    and n.value.func.id=='_apply_rotary' and len(n.value.args)==3)
        if match(a) and match(b) and ast.dump(a.value.args[1:][0])==ast.dump(b.value.args[1:][0]) and ast.dump(a.value.args[2])==ast.dump(b.value.args[2]):
            new=ast.Assign(targets=[ast.Tuple(elts=[a.targets[0],b.targets[0]],ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id='_versioned_rope_pair',ctx=ast.Load()),
                    args=[a.value.args[0],b.value.args[0],*a.value.args[1:]],keywords=[]))
            fn.body[i:i+2]=[new]
            replacements+=1
            break
    if replacements!=1:return None
    ns['_versioned_rope_pair']=pair
    exec(compile(ast.fix_missing_locations(tree),'<versioned-rope-forward>','exec'),ns)
    return ns[function.__name__]

def install(model):
    if model.config.model_type!='sdar':raise ValueError('runtime adapter currently supports SDAR only')
    count=0
    for layer in model.model.layers:
        attn=layer.self_attn
        if hasattr(attn,'_versioned_rope_originals'):continue
        originals={}
        for name in ('forward','_sdar_prefill_dense_forward','_sdar_losa_dense_forward','_paper_losa_dense_forward'):
            original=getattr(attn,name,None)
            if original is None:continue
            clone=_clone(original)
            if clone is not None:
                originals[name]=original
                setattr(attn,name,types.MethodType(clone,attn))
                count+=1
        attn._versioned_rope_originals=originals
        attn._versioned_rope_enabled=True
    return count

def restore(model):
    for layer in model.model.layers:
        attn=getattr(layer,'self_attn',None)
        if attn is None:continue
        for name,original in getattr(attn,'_versioned_rope_originals',{}).items():setattr(attn,name,original)
        if hasattr(attn,'_versioned_rope_originals'):del attn._versioned_rope_originals
        attn._versioned_rope_enabled=False
