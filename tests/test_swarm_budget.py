"""Pre-dispatch limits tested without network or credentials."""
import io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.impossible_budget import BudgetGateway

class Handler:
 def __init__(self,agent='A'):
  raw=json.dumps({'model':'openai/gpt-5','messages':[{'role':'user','content':'hello'}],'max_completion_tokens':8192}).encode()
  self.path='/api/v1/chat/completions';self.headers={'Content-Length':str(len(raw)),'x-agent-id':agent};self.rfile=io.BytesIO(raw);self.wfile=io.BytesIO();self.status=None
 def send_response(self,status):self.status=status
 def send_header(self,*args):pass
 def end_headers(self):pass

class Response:
 status=200
 def __init__(self,cost=.07,tokens=9000):self.body=json.dumps({'id':'fake','usage':{'cost':cost,'total_tokens':tokens},'choices':[]}).encode()
 def read(self,*args):return self.body
 def __enter__(self):return self
 def __exit__(self,*args):pass

class SwarmBudgetTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def gateway(self,**kwargs):return BudgetGateway('dummy-test-key',self.out,20,agent_limits={'A':.15,'B':2},**kwargs)
 def test_agent_quota_does_not_starve_peer(self):
  g=self.gateway()
  with patch('experiments.impossible_budget.urllib.request.urlopen',return_value=Response()) as upstream:
   a=Handler();g._handle_post(a);self.assertEqual(a.status,200)
   blocked=Handler();g._handle_post(blocked);self.assertEqual(blocked.status,402)
   b=Handler('B');g._handle_post(b);self.assertEqual(b.status,200)
   self.assertEqual(upstream.call_count,2)
  snap=g.snapshot();self.assertEqual(snap['agents']['A']['stop_reason'],'dollars');self.assertAlmostEqual(snap['agents']['B']['spent_usd'],.07);self.assertEqual(snap['reserved_tokens'],0)
 def test_token_limit_stops_before_second_dispatch(self):
  g=self.gateway(max_tokens=19000)
  with patch('experiments.impossible_budget.urllib.request.urlopen',return_value=Response()) as upstream:
   h=Handler('B');g._handle_post(h);self.assertEqual(h.status,200)
   h=Handler('B');g._handle_post(h);self.assertEqual(h.status,402);self.assertEqual(upstream.call_count,1)
  self.assertEqual(g.snapshot()['stop_reason'],'tokens');self.assertEqual(g.snapshot()['spent_tokens'],9000)
 def test_unknown_agent_and_expired_deadline_never_dispatch(self):
  g=self.gateway(deadline_monotonic=time.monotonic()-1)
  with patch('experiments.impossible_budget.urllib.request.urlopen') as upstream:
   h=Handler('stranger');g._handle_post(h);self.assertEqual(h.status,400)
   h=Handler();g._handle_post(h);self.assertEqual(h.status,402);upstream.assert_not_called()
  self.assertEqual(g.snapshot()['stop_reason'],'deadline')
 def test_uncertain_reservation_survives_restart_with_agent_identity(self):
  g=self.gateway()
  with patch('experiments.impossible_budget.urllib.request.urlopen',side_effect=TimeoutError('test timeout')):
   h=Handler();g._handle_post(h);self.assertEqual(h.status,502)
  before=g.snapshot();self.assertTrue(before['accounting_uncertain']);self.assertGreater(before['reserved_tokens'],0)
  restored=self.gateway();self.assertEqual(restored.snapshot()['agents']['A']['reserved_usd'],before['agents']['A']['reserved_usd'])
  with patch('experiments.impossible_budget.urllib.request.urlopen') as upstream:
   restored._handle_post(Handler('B'));upstream.assert_not_called()
 def test_dollar_limit_refusal_is_persisted(self):
  g=BudgetGateway('dummy',self.out,.55)
  with patch('experiments.impossible_budget.urllib.request.urlopen') as upstream:
   h=Handler();g._handle_post(h);self.assertEqual(h.status,402);upstream.assert_not_called()
  self.assertEqual(BudgetGateway('dummy',self.out,.55).snapshot()['stop_reason'],'dollars')


 def test_provider_usage_exceeding_reservation_stops_future_dispatch(self):
  g=self.gateway()
  with patch('experiments.impossible_budget.urllib.request.urlopen',return_value=Response(cost=.12,tokens=25000)) as upstream:
   h=Handler();g._handle_post(h);self.assertEqual(h.status,200)
   h=Handler('B');g._handle_post(h);self.assertEqual(h.status,402);self.assertEqual(upstream.call_count,1)
  self.assertEqual(g.snapshot()['stop_reason'],'reservation_exceeded')
 def test_invalid_stored_tokens_fail_closed(self):
  (self.out/'usage.jsonl').write_text(json.dumps({'model':'openai/gpt-5','id':'fake','agent':'A','cost':.01,'tokens':-1,'time':'test'})+'\n')
  g=self.gateway();self.assertTrue(g.snapshot()['accounting_uncertain'])
  with patch('experiments.impossible_budget.urllib.request.urlopen') as upstream:
   g._handle_post(Handler());upstream.assert_not_called()
 def test_unknown_stored_agent_fails_closed(self):
  (self.out/'usage.jsonl').write_text(json.dumps({'model':'openai/gpt-5','id':'fake','agent':'unknown','cost':.01,'tokens':1,'time':'test'})+'\n')
  self.assertTrue(self.gateway().snapshot()['accounting_uncertain'])

if __name__=='__main__':unittest.main()
