// modal, toast, and every button on every page
const $ = (s, r = document) => r.querySelector(s);
function toast(msg, kind) { const t = document.createElement('div'); t.className = 'toast ' + (kind || ''); t.textContent = msg; $('#toasts').append(t); setTimeout(() => t.remove(), 3200); }
function modal(title, body, foot) { $('#m-title').textContent = title; $('#m-body').innerHTML = body; $('#m-foot').innerHTML = foot || '<button class="btn" data-act="close">Close</button>'; $('#scrim').classList.add('on'); }
function closeModal() { $('#scrim').classList.remove('on'); }
async function call(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(m); }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}
const post = (url, data) => call(url, { method: 'POST', body: data instanceof FormData ? data : new URLSearchParams(data) });
const del = url => call(url, { method: 'DELETE' });
const field = (id, label, inner) => `<div class="field"><label for="${id}">${label}</label>${inner}</div>`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function setArmed(state) {
  const strip = $('#strip'); strip.dataset.armed = state.key;
  $('#armed-label').textContent = state.label; $('#strip .dot').classList.toggle('warn', state.key !== 'armed');
  $('#arm-buttons').innerHTML = state.key === 'armed'
    ? '<button class="btn sm" data-act="mute">Mute 1 h</button><button class="btn sm" data-act="disarm">Disarm</button>'
    : state.key === 'nobot' ? '' : '<button class="btn sm primary" data-act="arm">Arm</button>';
}

const acts = {
  close: closeModal,
  async mute() { setArmed(await post('/arm', { state: 'mute', minutes: 60 })); toast('Muted for 1 hour. Detection keeps running.'); },
  async disarm() { setArmed(await post('/arm', { state: 'disarm' })); toast('Disarmed. No alerts until you arm again.'); },
  async arm() { setArmed(await post('/arm', { state: 'arm' })); toast('Armed.', 'ok'); },
  async snapshot() { const r = await post('/snapshot', {}); toast(r.sent_to ? `Snapshot saved and sent to ${r.sent_to} recipient(s).` : 'Snapshot saved.', 'ok'); },
  async reset() { await post('/reset-tracking', {}); toast('Tracking reset: everyone in frame is re-identified on the next frame.'); },
  async detection(el) { const r = await post('/detection', { on: el.dataset.on }); location.reload(); },
  async pick() {
    let f; try { f = await call('/frame.json'); } catch (e) { return toast('No frame: detection is not running.', 'bad'); }
    if (!f.boxes.length) return toast('Nobody in frame right now.', 'bad');
    const boxes = f.boxes.map((b, i) => `<rect class="box${i ? '' : ' sel'}" data-i="${i}" x="${b.bbox[0]}" y="${b.bbox[1]}" width="${b.bbox[2] - b.bbox[0]}" height="${b.bbox[3] - b.bbox[1]}" fill="none" stroke="${b.kind === 'human' ? '#e5604f' : '#4dbf7f'}" stroke-width="3"/>`).join('');
    modal('Enrol from this frame', `<p style="margin:0;color:var(--ink-2)">Tap the box to enrol. The crop is saved as one photo; add more later for better matching.</p>
      <div class="pick"><div class="video" style="aspect-ratio:${f.width}/${f.height}"><svg viewBox="0 0 ${f.width} ${f.height}"><image href="/frame.jpg?${Date.now()}" width="${f.width}" height="${f.height}"/>${boxes}</svg></div></div>
      <div class="form">${field('p-name', 'Name', '<input id="p-name" placeholder="Name">')}<div class="field"><label>Enrol as</label><div id="p-kind" class="mono" style="padding:7px 0"></div></div></div>`,
      '<button class="btn" data-act="close">Cancel</button><button class="btn primary" data-act="pick-done">Enrol</button>');
    const show = i => { const b = f.boxes[i]; $('#p-kind').textContent = b.kind === 'human' ? 'person' : `pet · ${b.label}`; $('#m-body').dataset.i = i; };
    $('#m-body').querySelectorAll('rect.box').forEach(r => r.addEventListener('click', () => { $('#m-body').querySelectorAll('rect.box').forEach(x => x.classList.remove('sel')); r.classList.add('sel'); show(+r.dataset.i); }));
    $('#m-body')._frame = f; show(0);
  },
  async 'pick-done'() {
    const f = $('#m-body')._frame, b = f.boxes[+$('#m-body').dataset.i], name = $('#p-name').value.trim();
    if (!name) return toast('Give them a name first.', 'bad');
    await post('/enrol-from-frame', { kind: b.kind, name, bbox: JSON.stringify(b.bbox), class_id: b.class_id ?? '' });
    closeModal(); toast(`${name} enrolled. Roster reloaded.`, 'ok'); if (location.pathname === '/people') location.reload();
  },
  'enrol-person'() { modal('Enrol a person', `<form id="enrol-form" method="post" action="/enrol/human" enctype="multipart/form-data"><div class="form">${field('e-name', 'Name', '<input id="e-name" name="name" required placeholder="Hui">')}${field('e-photos', 'Photos of their face', '<input id="e-photos" name="photos" type="file" accept="image/*" multiple required>')}</div><div class="help" style="font-size:12px;color:var(--ink-3)">One clear face is enough; two or three from different angles is better. Or use <b>Enrol from this frame</b> on Live.</div></form>`, '<button class="btn" data-act="close">Cancel</button><button class="btn primary" data-act="submit-form">Enrol</button>'); },
  'enrol-pet'(el) { modal('Enrol a pet', `<form id="enrol-form" method="post" action="/enrol/animal" enctype="multipart/form-data"><div class="form">${field('e-name', 'Name', '<input id="e-name" name="name" required placeholder="Jacky">')}${field('e-kind', 'Kind', `<select id="e-kind" name="class_id">${el.dataset.classes}</select>`)}${field('e-photos', 'Three or more photos', '<input id="e-photos" name="photos" type="file" accept="image/*" multiple required>')}</div><div class="help" style="font-size:12px;color:var(--ink-3)">Photos from this camera work best — same angle, same light.</div></form>`, '<button class="btn" data-act="close">Cancel</button><button class="btn primary" data-act="submit-form">Enrol</button>'); },
  'submit-form'() { const f = $('#enrol-form'); if (f.reportValidity()) f.submit(); },
  async photos(el) {
    const id = el.dataset.id, r = await call(`/roster/${id}/photos`);
    modal(`${r.name} — photos`, `<div class="gal">${r.photos.map(p => `<div style="background:url(${p}) center/cover"></div>`).join('')}</div>
      <form id="enrol-form" method="post" action="/roster/${id}/photos" enctype="multipart/form-data">${field('a-photos', 'Add photos', '<input id="a-photos" name="photos" type="file" accept="image/*" multiple required>')}</form>`,
      '<button class="btn" data-act="close">Close</button><button class="btn primary" data-act="submit-form">Add</button>');
  },
  async testrec(el) {
    let r; try { r = await post(`/roster/${el.dataset.id}/test`, {}); } catch (e) { return toast(e.message, 'bad'); }
    const who = el.dataset.name, pet = el.dataset.kind === 'animal';
    modal(`Test — ${who}`, `<p style="margin:0">Ran ${pet ? 'pet re-ID' : 'face recognition'} on the current frame.</p>
      <div class="score"><i style="width:${Math.min(100, r.score * 100)}%${r.match ? '' : ';background:var(--bad)'}"></i><b style="left:${r.threshold * 100}%"></b></div>
      <div style="display:flex;justify-content:space-between;font:12px var(--mono);color:var(--ink-3)"><span>similarity <b style="color:var(--ink)">${r.score.toFixed(2)}</b></span><span>threshold ${r.threshold}</span></div>
      <p style="margin:0;color:var(--ink-2)">${r.match ? `Would be recognised as <b>${esc(who)}</b>.` : `Would <b>not</b> be recognised — no ${pet ? 'matching animal' : 'matching face'} in frame right now.`}</p>`);
  },
  forget(el) { modal(`Forget ${el.dataset.name}?`, '<p style="margin:0">Their photos are deleted and the next time they appear it is an intruder alert.</p>', `<button class="btn" data-act="close">Keep</button><button class="btn danger" data-act="forget-done" data-id="${el.dataset.id}">Forget</button>`); },
  async 'forget-done'(el) { await del(`/roster/${el.dataset.id}`); closeModal(); location.reload(); },
  'add-camera'() { modal('Add camera', camForm({}), '<button class="btn" data-act="close">Cancel</button><button class="btn" data-act="testform">Test</button><button class="btn primary" data-act="submit-form">Add</button>'); },
  editcam(el) { const d = el.dataset; modal(`Edit — ${d.name || 'camera ' + d.id}`, camForm(d), '<button class="btn" data-act="close">Cancel</button><button class="btn" data-act="testform">Test</button><button class="btn primary" data-act="submit-form">Save</button>'); },
  async testform() { const f = $('#enrol-form'); $('#test-out').textContent = 'testing…'; $('#test-out').className = 'pill'; try { const r = await post('/cameras/test', new FormData(f)); $('#test-out').textContent = r.message; $('#test-out').className = 'pill ' + (r.ok ? 'ok' : 'bad'); } catch (e) { $('#test-out').textContent = e.message; $('#test-out').className = 'pill bad'; } },
  async testcam(el) { const out = el.parentElement.parentElement.querySelector('.test-out') || el.nextElementSibling; out.textContent = 'testing…'; out.className = 'pill'; const r = await post(`/cameras/${el.dataset.id}/test`, {}); out.textContent = r.message; out.className = 'pill ' + (r.ok ? 'ok' : 'bad'); },
  removecam(el) { modal('Remove camera?', `<p style="margin:0"><b>${esc(el.dataset.name)}</b> stops streaming; its past events stay.</p>`, `<button class="btn" data-act="close">Keep</button><button class="btn danger" data-act="removecam-done" data-id="${el.dataset.id}">Remove</button>`); },
  async 'removecam-done'(el) { await del(`/cameras/${el.dataset.id}`); closeModal(); location.reload(); },
  'add-recipient'() { modal('Add recipient', `<form id="enrol-form" method="post" action="/telegram"><div class="form">${field('r-chat', 'Chat ID', '<input id="r-chat" name="chat_id" required inputmode="numeric" placeholder="128402991">')}${field('r-name', 'Name', '<input id="r-name" name="username" placeholder="optional">')}</div><label class="sw"><input type="checkbox" name="humans" value="true" checked> alert on people</label><label class="sw"><input type="checkbox" name="animals" value="true" checked> alert on animals</label><div class="help" style="font-size:12px;color:var(--ink-3)">Message the bot first, then get your chat id from @userinfobot. A hello is sent to confirm.</div></form>`, '<button class="btn" data-act="close">Cancel</button><button class="btn primary" data-act="submit-form">Add</button>'); },
  async testmsg(el) { try { const r = await post(`/telegram/${el.dataset.id}/test`, {}); toast(r.ok ? `Test message sent to ${el.dataset.name} ✓` : 'Telegram refused the message.', r.ok ? 'ok' : 'bad'); } catch (e) { toast(e.message, 'bad'); } },
  removerec(el) { modal('Stop alerting?', `<p style="margin:0"><b>${esc(el.dataset.name)}</b> will get no more alerts and cannot command the bot.</p>`, `<button class="btn" data-act="close">Keep</button><button class="btn danger" data-act="removerec-done" data-id="${el.dataset.id}">Remove</button>`); },
  async 'removerec-done'(el) { await del(`/telegram/${el.dataset.id}`); closeModal(); location.reload(); },
  async 'save-delivery'(el) { await post('/telegram/delivery', new FormData(el.closest('form'))); toast('Saved. Applied without a restart.', 'ok'); },
  async 'save-settings'(el) { await post('/settings', new FormData(el.closest('form'))); toast('Saved. Applied without a restart.', 'ok'); },
  async event(el) {
    const e = await call(`/events/${el.dataset.id}`), unknown = /^Unknown/.test(e.entity_name || '');
    modal(e.entity_name || 'Unknown', `${e.photo ? `<img src="${e.photo}" style="width:100%;border-radius:4px;background:#000" alt="">` : '<div class="bigph"></div>'}
      <p style="margin:0;color:var(--ink-2)">${e.caption ? esc(e.caption) : unknown ? 'No caption (VLM off or still working).' : 'Family — logged, no alert sent.'}</p>
      <div style="font:12px var(--mono);color:var(--ink-3)">${e.detected_at} · ${e.detection_type} · ${Math.round((e.confidence || 0) * 100)} % · ${e.notification_sent ? 'alert sent' : 'no alert'}</div>`,
      unknown && e.photo ? '<button class="btn" data-act="close">Close</button><button class="btn primary" data-act="pick">This is family — enrol from the live frame…</button>' : '<button class="btn" data-act="close">Close</button>');
  },
};
function camForm(d) {
  return `<form id="enrol-form" method="post" action="/cameras${d.id ? '/' + d.id : ''}"><div class="form">${field('c-name', 'Name', `<input id="c-name" name="name" value="${esc(d.name || '')}" placeholder="Front gate">`)}
    <div class="field" style="grid-column:1/-1"><label for="c-url">Stream URL</label><input id="c-url" name="url" value="${esc(d.url || '')}" placeholder="rtsp://user:pw@192.168.1.20:554/stream2"><span class="help">Or fill the parts below. Credentials go in the URL.</span></div>
    ${field('c-proto', 'Protocol', '<select id="c-proto" name="protocol"><option>rtsp</option><option>http</option><option>https</option></select>')}${field('c-host', 'Host', '<input id="c-host" name="host" placeholder="192.168.1.20">')}
    ${field('c-port', 'Port', '<input id="c-port" name="port" value="0" title="0 = protocol default">')}${field('c-path', 'Path', '<input id="c-path" name="path" placeholder="/stream2, /video for DroidCam">')}
    ${field('c-user', 'Username', '<input id="c-user" name="username">')}${field('c-pass', 'Password', '<input id="c-pass" name="password" type="password">')}
    <div class="field"><label>&nbsp;</label><label class="sw"><input type="checkbox" name="auto" value="true" ${d.auto === 'false' ? '' : 'checked'}> Connect on startup</label></div></div>
    <div style="margin-top:8px"><span class="pill" id="test-out">not tested</span></div></form>`;
}
document.addEventListener('click', ev => {
  const el = ev.target.closest('[data-act]'); if (!el) return;
  if (el.dataset.act === 'event' && ev.target.closest('button,a')) return;
  ev.preventDefault();
  Promise.resolve(acts[el.dataset.act]?.(el)).catch(e => toast(e.message || 'That failed.', 'bad'));
});
document.addEventListener('change', async ev => {
  const t = ev.target;
  if (t.matches('.toggles input')) { const f = new FormData(); $('.toggles').querySelectorAll('input').forEach(i => f.set(i.name, i.checked)); await post('/toggles', f); toast(`${t.parentElement.textContent.trim()} ${t.checked ? 'on' : 'off'}`); }
  if (t.matches('[data-prefs]')) { const row = t.closest('[data-chat]'); const f = new FormData(); row.querySelectorAll('[data-prefs]').forEach(i => f.set(i.dataset.prefs, i.checked)); await post(`/telegram/${row.dataset.chat}/prefs`, f); toast('Saved.', 'ok'); }
  if (t.name === 'tier') { await post('/tier', { tier: t.value }); toast(`Switching to ${t.value} tier — reloading models (~10 s)…`); document.querySelectorAll('label.tierbox').forEach(l => l.classList.toggle('on', l.contains(t))); }
  if (t.matches('#cam-pick')) { location.href = '/?camera=' + t.value; }
});
document.querySelectorAll('.range').forEach(r => { const i = r.querySelector('input'), o = r.querySelector('output'); const fmt = i.dataset.fmt || '{}'; const show = () => o.textContent = fmt.replace('{}', i.step && i.step.includes('.') ? (+i.value).toFixed(2) : i.value); i.addEventListener('input', show); show(); });
$('#scrim').addEventListener('click', e => { if (e.target.id === 'scrim') closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
