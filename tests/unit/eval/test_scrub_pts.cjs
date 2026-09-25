// Unit-test the component protocol, without launching/controlling a browser.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('src/mfe/browser/video_scrub/frontend/index.html', 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function fixture(supports = true) {
  const events = {}, clicks = {}, messages = [];
  let callback;
  const v = {paused: true, seeking: false, currentTime: 40, dataset: {},
    addEventListener: (key, fn) => {events[key] = fn;},
    removeEventListener: () => {}};
  if (supports) {
    v.requestVideoFrameCallback = fn => {callback = fn; return 1;};
    v.cancelVideoFrameCallback = () => {callback = null;};
  }
  const elements = {v, pick: {addEventListener: (key, fn) => {clicks[key] = fn;}}};
  const document = {getElementById: id => elements[id] ||= {}, documentElement: {scrollHeight: 400}};
  const window = {parent: {postMessage: value => messages.push(value)}, addEventListener: () => {}};
  vm.runInNewContext(code, {document, window, Date});
  return {v, events, clicks, elements, messages, frame: pts => callback(0, {mediaTime: pts})};
}
const f = fixture();
f.clicks.click();
assert.equal(f.messages.filter(m => m.type === 'streamlit:setComponentValue').length, 0);
f.frame(39.731);
f.clicks.click();
let selected = f.messages.find(m => m.type === 'streamlit:setComponentValue').value;
assert.equal(selected.seconds, 39.731); // NOT currentTime=40 or frame/FPS
assert.equal(selected.time_source, 'video_frame_callback');
f.v.seeking = true;
f.events.seeking();
assert.equal(f.elements.pick.disabled, true);
f.v.seeking = false;
f.clicks.click();
assert.equal(f.messages.filter(m => m.type === 'streamlit:setComponentValue').length, 1);
f.frame(80.123);
f.clicks.click();
assert.equal(f.messages.at(-1).value.seconds, 80.123);
const unsupported = fixture(false);
unsupported.events.loadedmetadata();
assert.equal(unsupported.elements.pick.disabled, true);
unsupported.clicks.click();
assert.equal(unsupported.messages.filter(m => m.type === 'streamlit:setComponentValue').length, 0);
console.log('PTS protocol tests passed: VFR, stale seek, unsupported browser, empty frame.');
