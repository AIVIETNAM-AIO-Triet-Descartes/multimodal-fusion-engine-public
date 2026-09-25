"""Render the real Streamlit entry point with offline engine fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from mfe.contracts.tools import Candidate
from mfe.eval.query_package import PackagedQuery

APP = Path(__file__).resolve().parents[3] / 'app' / 'submit_lab.py'


@pytest.mark.parametrize('kind', ['KIS', 'QA', 'TRAKE', 'EMPTY'])
def test_render_and_download_bytes(kind):
    qtype = 'KIS' if kind == 'EMPTY' else kind
    query = PackagedQuery('query-test-' + qtype.lower(), qtype, 'mô tả', Path('test.txt'),
                          events=('a', 'b') if qtype == 'TRAKE' else ())
    candidates = [Candidate(keyframe_id=f'L21_V001#{n:06d}', video_id='L21_V001',
                            score=1, evidence_id=f'siglip:L21_V001:{n}') for n in (100, 200)]
    rows = ([NS(video_id='L21_V001', frame_ids=[100, 200]),
             NS(video_id='L22_V002', frame_ids=[400, 800])]
            if qtype == 'TRAKE' else candidates)
    if kind == 'EMPTY':
        rows = []
    db = NS(get_keyframe=lambda key: NS(video_id='L21_V001', pts_time=29.701,
                                        frame_idx=int(key.split('#')[1])),
            get_video=lambda _: NS(fps=25))
    captured = []
    original_download = st.download_button

    def download(*args, **kwargs):
        captured.append(kwargs)
        return original_download(*args, **kwargs)

    st.cache_resource.clear()
    with patch('mfe.online.flat_arms.load_flat_bundle', return_value=None), \
         patch('mfe.online.flat_arms.build_flat_cascade', return_value=NS(arms=[])), \
         patch('mfe.mapping.relational.RelationalMap', return_value=db), \
         patch('mfe.online.keyframe_images.KeyframeImages') as images, \
         patch('mfe.online.video_source.resolve_video', return_value=None), \
         patch('streamlit.download_button', side_effect=download):
        images.return_value.cached.return_value = None
        at = AppTest.from_file(str(APP), default_timeout=30)
        at.session_state['parsed'] = [query]
        at.session_state['results'] = {query.query_id: rows}
        at.session_state['order_' + query.query_id] = [c.keyframe_id for c in reversed(candidates)]
        at.session_state['answer_keep_' + query.query_id] = 'áo đỏ'
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        if kind == 'EMPTY':
            assert not captured
            assert any('Chưa có ứng viên' in w.value for w in at.warning)
        else:
            assert len(captured) == 1
            assert captured[0]['file_name'].endswith('.json')
            payload = json.loads(captured[0]['data'])
            answers = payload['answerSets'][0]['answers']
            assert len(answers) == 1
            if kind == 'KIS':
                assert answers[0]['start'] == answers[0]['end'] == 29701
            elif kind == 'QA':
                assert answers[0]['text'] == 'QA-áo đỏ-L21_V001-29701'
            else:
                assert answers[0]['text'] == 'TR-L21_V001-100,200'
                captured.clear()
                at.selectbox(key='trake_top_' + query.query_id).select(1).run()
                assert not at.exception
                selected = json.loads(captured[-1]['data'])['answerSets'][0]['answers'][0]
                assert selected['text'] == 'TR-L22_V002-400,800'
    st.cache_resource.clear()
