"""Auth/scope validation for actual cockpit route, with unit-test client fixtures."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def client_and_calls():
    path=Path(__file__).resolve().parents[3]/'web/server.py'
    spec=importlib.util.spec_from_file_location('dres_logging_test_server',path)
    server=importlib.util.module_from_spec(spec);sys.modules[spec.name]=server;spec.loader.exec_module(server)
    calls=[]
    class LocalClient:
        def log_query(self,eid,payload):calls.append(('query',eid,payload));return {'status':True}
        def log_results(self,eid,payload):calls.append(('result',eid,payload));return {'status':True}
    import threading
    manager=SimpleNamespace(dres_lock=threading.RLock(),dres=lambda:LocalClient(),
        shared_state={'status':'done','query_type':'KIS','results':[{'video_id':'shared','timestamp_ms':1}]},
        private_states={'operator':{'status':'done','query_type':'KIS','results':[{'video_id':'own','timestamp_ms':2}]},
                        'other':{'status':'done','query_type':'KIS','results':[{'video_id':'other','timestamp_ms':3}]}})
    async def login(request):request.session.update(user='operator',csrf='csrf-test');return JSONResponse({'ok':True})
    app=Starlette(routes=[Route('/login-test',login),Route('/api/dres/log',server.dres_log,methods=['POST'])],
                  middleware=[Middleware(SessionMiddleware,secret_key='unit-test-secret')])
    app.state.manager=manager
    return TestClient(app),calls


def payload(**changes):
    return {'evaluation_id':'eval-1','scope':'private','include_results':True,
            'events':[{'category':'TEXT','type':'query','value':'scene'}],**changes}


def test_logging_requires_auth_and_csrf():
    c,calls=client_and_calls()
    assert c.post('/api/dres/log',json=payload()).status_code==401
    c.get('/login-test')
    assert c.post('/api/dres/log',json=payload()).status_code==403
    assert calls==[]


def test_private_results_come_from_authenticated_owner():
    c,calls=client_and_calls();c.get('/login-test')
    r=c.post('/api/dres/log',headers={'X-CSRF-Token':'csrf-test'},json=payload(user='other',results=[{'video_id':'injected','timestamp_ms':9}]))
    assert r.status_code==200
    result=next(p for kind,_,p in calls if kind=='result')
    assert result['results'][0]['answer']['mediaItemName']=='own'


def test_bad_json_scope_or_event_cannot_reach_upstream():
    c,calls=client_and_calls();c.get('/login-test');headers={'X-CSRF-Token':'csrf-test'}
    assert c.post('/api/dres/log',headers=headers,json=[]).status_code==400
    assert c.post('/api/dres/log',headers=headers,json=payload(scope=[])).status_code==400
    assert c.post('/api/dres/log',headers=headers,json=payload(events=[None])).status_code==400
    assert calls==[]
