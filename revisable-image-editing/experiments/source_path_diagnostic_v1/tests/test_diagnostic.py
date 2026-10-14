import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_path import common as c
from source_path.attention import key_bias, combine_mask, Intervention
from source_path.data import Exclusions
from source_path.native import parse_proposal, parse_native_json
from source_path.scoring import Ledger, cost
from source_path.statistics import schedule, parse_judgment, estimate, summarize
from source_path.worker import candidate_definitions
from source_path.runtime import require_allocation


class Contracts(unittest.TestCase):
    def test_config_fixed_denominators(self):
        cfg=c.config(c.PACKAGE/'config.yaml')
        self.assertEqual(cfg['cases_per_family']*len(c.FAMILIES),256)
        self.assertEqual(cfg['revision'],c.REVISION)

    def test_corrupted_seal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'; c.seal(p,{'a':1}); value=json.loads(p.read_text()); value['a']=2
            p.write_text(json.dumps(value))
            with self.assertRaises(ValueError): c.read(p)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): c.asset(d,'../secret.env')

    def test_duplicate_worker_is_refused_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as d:
            with c.execution_lock(d,'generate'):
                with self.assertRaises(RuntimeError):
                    with c.execution_lock(d,'score'): pass
            with c.execution_lock(d,'score'): pass

    def test_noncompensatory_checks(self):
        value={k:True for k in c.CHECKS}; value['requested_change']='unknown'
        self.assertEqual(c.require_native_checks(value)['requested_change'],'unknown')
        value['requested_change']=1; self.assertIsNone(c.require_native_checks(value))

    def test_selection_uses_first_passing_and_no_external_scores(self):
        pool=['base42','a','b']; receipts={k:{'accepted':False,'external_score':999} for k in pool}
        receipts['a']['accepted']=True; receipts['b']['accepted']=True
        self.assertEqual(c.choose(pool,receipts),('a',True))
        receipts['a']['accepted']=False; receipts['b']['accepted']=False
        self.assertEqual(c.choose(pool,receipts),('base42',False))

    def test_proposal_rejects_noop_and_schema_failure(self):
        value={'eligible':True,'instruction':'recolor the car','subject':'car','source_state':'red',
               'target_state':'blue','preserve':['shape']}
        self.assertIsNotNone(parse_proposal(value,'attribute'))
        value['target_state']='red'; self.assertIsNone(parse_proposal(value,'attribute'))
        value['eligible']=1; self.assertIsNone(parse_proposal(value,'attribute'))

    def test_native_duplicate_and_nonfinite_fields_abstain(self):
        self.assertEqual(parse_native_json('{"eligible":false,"eligible":true}'),{'unknown':True})
        self.assertEqual(parse_native_json('{"requested_change":NaN}'),{'unknown':True})

    def test_equal_candidate_budget(self):
        values=candidate_definitions({'prompt':'change background'},c.config(c.PACKAGE/'config.yaml'))
        self.assertEqual(len(values),10); self.assertEqual(len({r[0] for r in values}),10)
        pools=[['base42','semantic42','reference42','both42'],['base42','base101','base202','base303'],
               ['base42','clarify101','clarify202','clarify303']]
        for pool in pools:
            self.assertEqual(len(pool),4); self.assertTrue(set(pool)<=set(r[0] for r in values))
        self.assertEqual([r[2] for r in values[:4]],[42]*4)

    def test_exclusion_detects_perceptual_near_duplicate(self):
        a={'encoded_sha256':'a','decoded_rgb_sha256':'b','dhash64':'0000000000000000'}
        exclusion=Exclusions([a])
        self.assertTrue(exclusion.contains({'encoded_sha256':'c','decoded_rgb_sha256':'d','dhash64':'000000000000000f'}))
        self.assertFalse(exclusion.contains({'encoded_sha256':'c','decoded_rgb_sha256':'d','dhash64':'ffffffffffffffff'}))


class Attention(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch

    def test_bias_targets_only_declared_keys(self):
        torch=self.torch; visual=torch.tensor([[False,True,False]])
        bias=key_bias(visual,5,2,.5,.5,torch=torch)
        expected=torch.tensor([0,math.log(.5),0,0,0,0,math.log(.5),math.log(.5)])
        self.assertTrue(torch.allclose(bias.flatten(),expected))

    def test_padding_cannot_be_unmasked(self):
        torch=self.torch
        old=torch.tensor([[[[True,False,True,True,True]]]])
        result=combine_mask(old,torch.zeros(1,1,1,5),torch=torch)
        self.assertTrue(torch.isneginf(result[...,1]).all())

    def test_scale_is_attention_mass_not_embedding_scale(self):
        torch=self.torch; from torch.nn.functional import scaled_dot_product_attention as sdpa
        q=torch.zeros(1,1,1,5); k=torch.zeros(1,1,5,5); v=torch.eye(5)[None,None]
        bias=key_bias(torch.tensor([[True,False]]),3,1,.5,.5,torch=torch)
        output=sdpa(q,k,v,attn_mask=bias).flatten()
        self.assertAlmostEqual(float(output[0]/output[1]),.5,places=6)
        self.assertAlmostEqual(float(output[-1]/output[1]),.5,places=6)
        self.assertAlmostEqual(float(output.sum()),1.,places=6)

    def test_unity_hook_does_not_modify_kwargs(self):
        intervention=object.__new__(Intervention)
        intervention.calls=0; intervention.semantic_scale=intervention.reference_scale=1.
        kwargs={'attention_mask':object()}; previous=kwargs.copy()
        self.assertIsNone(intervention.hook(None,(),kwargs)); self.assertEqual(kwargs,previous)

    def test_cfg_uses_separate_visual_masks(self):
        torch=self.torch; intervention=object.__new__(Intervention)
        intervention.semantic_scale=.5; intervention.reference_scale=1.; intervention.reference_length=1
        intervention.changed_calls=0; intervention.visual_masks=[torch.tensor([[True,False]]),torch.tensor([[False,True]])]
        kwargs={'hidden_states':torch.zeros(1,3,2),'encoder_hidden_states':torch.zeros(1,2,2)}
        intervention.calls=0; _,positive=intervention.hook(None,(),kwargs)
        intervention.calls=60; _,negative=intervention.hook(None,(),kwargs)
        self.assertLess(float(positive['attention_mask'][0,0,0,0]),0)
        self.assertEqual(float(positive['attention_mask'][0,0,0,1]),0)
        self.assertEqual(float(negative['attention_mask'][0,0,0,0]),0)
        self.assertLess(float(negative['attention_mask'][0,0,0,1]),0)

    def test_no_cuda_context_created(self):
        self.assertFalse(self.torch.cuda.is_initialized())


class Evaluation(unittest.TestCase):
    @staticmethod
    def cases(n=32):
        return [{'id':f'{f}_{i}','family':f,'prompt':'test','fingerprint':{'decoded_rgb_sha256':f'{f}_{i}'}}
                for f in c.FAMILIES for i in range(n)]

    def test_schedule_has_both_orders_and_identical_controls(self):
        tasks=schedule(self.cases()); self.assertEqual(len(tasks),4104)
        self.assertEqual(sum(r['control'] for r in tasks),8)
        pair=[r for r in tasks if r['pair']=='selected_vs_base' and r['case_id']=='add_0']
        self.assertEqual(len(pair),2); self.assertEqual(pair[0]['a_arm'],pair[1]['b_arm'])

    def test_duplicate_json_fields_rejected(self):
        with self.assertRaises(ValueError): parse_judgment('{"semantic":"A","semantic":"B"}')

    def test_macro_does_not_reward_large_categories(self):
        rows=[{'family':'a','source_group':str(i),'semantic':1.} for i in range(20)]
        rows += [{'family':'b','source_group':'z','semantic':0.}]
        result=estimate(rows,'semantic',['a','b'],32,42)
        self.assertEqual(result['mean'],.5)

    def synthetic(self, choice='tie'):
        cases=self.cases(8); tasks=schedule(cases); receipts=[]
        for task in tasks:
            answer='tie' if task['control'] else choice
            if choice=='first' and not task['control']:
                answer='A' if task['a_arm']==c.PAIR_ARMS[task['pair']][0] else 'B'
            receipts.append({'task':task,'judgment':{**{k:answer for k in c.FIELDS},'reason':'synthetic unit fixture'}})
        records=[{'selection':{a:{'accepted':True} for a in ('selected','resample','clarify')}} for _ in cases]
        cfg=c.config(c.PACKAGE/'config.yaml'); cfg['bootstrap_samples']=32
        return cases,tasks,receipts,records,cfg

    def test_all_ties_cannot_pass(self):
        result=summarize(*self.synthetic()); self.assertFalse(result['passed']); self.assertEqual(result['decision'],'do_not_train')

    def test_display_order_mapping(self):
        result=summarize(*self.synthetic('first')); self.assertTrue(result['passed'])
        self.assertEqual(result['statistics']['selected_vs_base']['macro']['semantic']['mean'],1.)

    def test_failed_identical_control_blocks_decision(self):
        args=list(self.synthetic()); next(r for r in args[2] if r['task']['control'])['judgment']['semantic']='A'
        with self.assertRaises(ValueError): summarize(*args)

    def test_unassessable_is_not_dropped(self):
        args=list(self.synthetic()); args[2][0]['judgment']['semantic']='unassessable'
        with self.assertRaises(ValueError): summarize(*args)

    def test_missing_order_blocks_decision(self):
        args=list(self.synthetic()); args[2].pop()
        with self.assertRaises(ValueError): summarize(*args)


class Budget(unittest.TestCase):
    def test_over_reservation_cost_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','x',4,1.); ledger.reserve('a')
            with self.assertRaises(RuntimeError): ledger.finish('a',.2,'failed')
            self.assertEqual(c.read(ledger.path)['entries']['a']['usd'],.2)

    def test_pending_request_is_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json'; ledger=Ledger(p,'x',4,1.); ledger.reserve('a')
            with self.assertRaises(RuntimeError): Ledger(p,'x',4,1.)
            with self.assertRaises(RuntimeError): ledger.reserve('a')

    def test_reservation_limit(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','x',4,.1); ledger.reserve('a')
            with self.assertRaises(RuntimeError): ledger.reserve('b')

    def test_concurrent_ledger_writes_are_serialized(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','x',8,2.)
            workers=[threading.Thread(target=lambda i=i:ledger.reserve(str(i))) for i in range(8)]
            for worker in workers: worker.start()
            for worker in workers: worker.join()
            self.assertEqual(len(c.read(ledger.path)['entries']),8)

    def test_usage_caching_accounting(self):
        self.assertAlmostEqual(cost({'prompt_tokens':1000,'completion_tokens':100,'prompt_tokens_details':{'cached_tokens':500}}),.00205)
        with self.assertRaises(ValueError): cost({'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'cached_tokens':2}})


class Runtime(unittest.TestCase):
    def test_direct_mode_refuses_original_shared_host(self):
        origin=json.loads((c.PACKAGE/'assets/source_machine.json').read_text())
        with mock.patch.dict('os.environ',{'SP_EXECUTION_MODE':'dedicated','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             mock.patch('source_path.runtime.socket.gethostname',return_value=origin['hostname']):
            with self.assertRaises(RuntimeError): require_allocation('gpu')

    def test_cpu_mode_refuses_visible_gpu(self):
        with mock.patch.dict('os.environ',{'SP_EXECUTION_MODE':'dedicated','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             mock.patch('source_path.runtime.socket.gethostname',return_value='new-dedicated-test-host'):
            with self.assertRaises(RuntimeError): require_allocation('cpu')

    def test_slurm_non_numeric_step_rejected_before_scheduler(self):
        with mock.patch.dict('os.environ',{'SP_EXECUTION_MODE':'slurm','SLURM_JOB_ID':'123','SLURM_STEP_ID':'extern','TMUX':'x'},clear=True), \
             mock.patch('source_path.runtime.subprocess.check_output') as scheduler:
            with self.assertRaises(RuntimeError): require_allocation('gpu')
            scheduler.assert_not_called()


if __name__=='__main__': unittest.main()
