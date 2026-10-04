import pytest
from mfe.eval.dres_client import DresClient, DresError, build_query_log, build_result_log, build_selected_payload


def test_log_uses_epoch_ms_and_preserves_temporal_coordinates():
    events=[{'category':'TEXT','type':'query','value':'scene'}]
    payload=build_result_log([{'video_id':'video_1','timestamp_ms':1500,'collection_name':'collection A'}],events,timestamp_ms=123456)
    assert payload['timestamp']==123456
    assert payload['sortType']=='retrieval-order'
    assert payload['events'][0]['timestamp']==123456
    assert payload['results']==[{'rank':1,'answer':{'mediaItemName':'video_1','start':1500,'end':1500,'mediaItemCollectionName':'collection A'}}]


@pytest.mark.parametrize('events', [None,{},[],[None],[{'category':[],'type':'q','value':'v'}],
                                  [{'category':'INVALID','type':'q','value':'v'}]])
def test_invalid_events_rejected_without_unhandled_errors(events):
    with pytest.raises(DresError):build_query_log(events)


def test_ranked_alignment_without_timestamp_cannot_be_fabricated_as_zero():
    with pytest.raises(DresError):
        build_result_log([{'video_id':'v','events':[]}],[{'category':'TEXT','type':'q','value':'v'}])


def test_named_collection_resolves_ambiguous_media(monkeypatch):
    monkeypatch.setenv('MFE_DRES_COLLECTION','V3C')
    answer=build_selected_payload('KIS',{'video_id':'video','timestamp_ms':123})['answerSets'][0]['answers'][0]
    assert answer['mediaItemCollectionName']=='V3C'
    answer=build_selected_payload('KIS',{'video_id':'video','timestamp_ms':123,'collection_name':'LHE'})['answerSets'][0]['answers'][0]
    assert answer['mediaItemCollectionName']=='LHE'


def test_client_log_endpoint_keeps_session_server_side(monkeypatch):
    calls=[]
    client=DresClient('test','secret',session_id='local-token')
    monkeypatch.setattr(client,'_request',lambda method,path,**kwargs:calls.append((method,path,kwargs)) or {'status':True})
    client.log_query('eval-1',{'timestamp':1,'events':[]})
    client.log_results('eval-1',{'timestamp':1})
    assert [c[1] for c in calls]==['/api/v2/log/query/eval-1','/api/v2/log/result/eval-1']
    assert all(c[2]['query']=={'session':'local-token'} for c in calls)
