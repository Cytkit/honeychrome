/* Honeychrome website behaviour: theme toggle, menus, download buttons that
   match the visitor's operating system, on-page contents, copy buttons,
   image zoom and documentation search. No dependencies. */
(function () {
  'use strict';

  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ }
  }

  /* Theme toggle */
  function currentTheme() {
    var set = document.documentElement.getAttribute('data-theme');
    if (set) { return set; }
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      store('hc-theme', next);
    });
  });

  /* Main menu and docs sidebar on small screens */
  function toggler(buttonSelector, targetSelector) {
    var btn = document.querySelector(buttonSelector);
    var target = document.querySelector(targetSelector);
    if (!btn || !target) { return; }
    btn.addEventListener('click', function () {
      var open = target.classList.toggle('is-open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }
  toggler('[data-nav-toggle]', '#site-nav');
  toggler('[data-sidebar-toggle]', '#docs-nav');

  /* Operating-system detection for download buttons */
  function detectOS() {
    var p = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    var ua = navigator.userAgent || '';
    if (/iPhone|iPad|Android/i.test(ua)) { return null; }
    if (/mac/i.test(p) || /Mac OS X/.test(ua)) { return 'mac'; }
    if (/win/i.test(p) || /Windows/.test(ua)) { return 'windows'; }
    if (/linux/i.test(p) || /Linux/.test(ua)) { return 'linux'; }
    return null;
  }
  var os = detectOS();
  var labels = { mac: 'macOS', windows: 'Windows', linux: 'Linux' };
  if (os) {
    document.querySelectorAll('[data-os-download]').forEach(function (a) {
      var label = a.querySelector('[data-os-label]');
      if (label) { label.textContent = 'Download for ' + labels[os]; }
      a.setAttribute('href', a.getAttribute('href').replace(/#.*$/, '') + '#' + os);
    });
    var card = document.getElementById(os);
    if (card && card.classList.contains('platform')) { card.classList.add('is-detected'); }
  }

  /* On-page contents for docs pages */
  var tocBox = document.querySelector('[data-toc]');
  var tocSource = document.querySelector('[data-toc-source]');
  if (tocBox && tocSource) {
    var heads = tocSource.querySelectorAll('h2[id], h3[id]');
    if (heads.length > 1) {
      var list = tocBox.querySelector('[data-toc-list]');
      var links = [];
      heads.forEach(function (h) {
        var li = document.createElement('li');
        li.className = 'toc-' + h.tagName.toLowerCase();
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        li.appendChild(a);
        list.appendChild(li);
        links.push({ a: a, h: h });
      });
      tocBox.hidden = false;
      var onScroll = function () {
        var active = links[0];
        links.forEach(function (l) { if (l.h.getBoundingClientRect().top < 120) { active = l; } });
        links.forEach(function (l) { l.a.classList.toggle('is-active', l === active); });
      };
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }
  }

  /* Copy buttons on code blocks */
  document.querySelectorAll('.prose pre').forEach(function (pre) {
    if (!navigator.clipboard) { return; }
    var holder = pre.closest('.highlighter-rouge') || pre;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText(pre.innerText.replace(/\n$/, '')).then(function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
      });
    });
    holder.appendChild(btn);
  });

  /* Click to enlarge figures */
  document.addEventListener('click', function (e) {
    var link = e.target.closest('[data-zoom]');
    if (!link || e.metaKey || e.ctrlKey) { return; }
    e.preventDefault();
    var overlay = document.createElement('div');
    overlay.className = 'zoom-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-label', 'Enlarged image');
    var img = document.createElement('img');
    img.src = link.getAttribute('href');
    var small = link.querySelector('img');
    img.alt = small ? small.alt : '';
    overlay.appendChild(img);
    var close = function () { overlay.remove(); document.removeEventListener('keydown', onKey); };
    var onKey = function (ev) { if (ev.key === 'Escape') { close(); } };
    overlay.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    document.body.appendChild(overlay);
  });

  /* Documentation search over an index written at build time
     (assets/js/search-index.js), so it also works from a saved copy. */
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function snippet(text, terms) {
    var lower = text.toLowerCase();
    var at = -1;
    terms.forEach(function (t) { var i = lower.indexOf(t); if (i >= 0 && (at < 0 || i < at)) { at = i; } });
    var start = Math.max(0, at - 60);
    var out = text.substr(start, 180);
    out = escapeHtml(out);
    terms.forEach(function (t) {
      out = out.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
    });
    return out;
  }
  /* A copy of the site saved to disk (tools/make_preview.py) records the
     path back to its root, so search results can link relatively. */
  var rootMeta = document.querySelector('meta[name="hc-root"]');
  function pageUrl(u) {
    if (!rootMeta || u.charAt(0) !== '/') { return u; }
    var parts = u.split('#');
    var path = parts[0].slice(1);
    if (path === '' || path.charAt(path.length - 1) === '/') { path += 'index.html'; }
    return rootMeta.getAttribute('content') + path + (parts[1] ? '#' + parts[1] : '');
  }
  function runSearch(q) {
    var index = window.HC_SEARCH_INDEX || [];
    var terms = q.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 1; });
    if (!terms.length) { return []; }
    return index.map(function (d) {
      var title = d.t.toLowerCase();
      var body = d.c.toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i];
        var inTitle = title.indexOf(t) >= 0;
        var count = body.split(t).length - 1;
        if (!inTitle && !count) { return null; }
        score += (inTitle ? 10 : 0) + Math.min(count, 20);
      }
      return { d: d, score: score };
    }).filter(Boolean).sort(function (a, b) { return b.score - a.score; }).slice(0, 8)
      .map(function (r) { return { url: pageUrl(r.d.u), title: r.d.t, text: snippet(r.d.c, terms) }; });
  }
  document.querySelectorAll('[data-search]').forEach(function (form) {
    var input = form.querySelector('[data-search-input]');
    var box = form.querySelector('[data-search-results]');
    var active = -1;
    function render() {
      var q = input.value.trim();
      active = -1;
      if (q.length < 2) { box.hidden = true; box.innerHTML = ''; return; }
      var hits = runSearch(q);
      box.innerHTML = hits.length ? hits.map(function (h) {
        return '<a href="' + h.url + '"><strong>' + escapeHtml(h.title) + '</strong><small>' + h.text + '</small></a>';
      }).join('') : '<p class="search__empty">No pages match “' + escapeHtml(q) + '”.</p>';
      box.hidden = false;
    }
    input.addEventListener('input', render);
    input.addEventListener('keydown', function (e) {
      var items = box.querySelectorAll('a');
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (!items.length) { return; }
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
        items.forEach(function (a, i) { a.classList.toggle('is-active', i === active); });
      } else if (e.key === 'Escape') {
        box.hidden = true;
      }
    });
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var items = box.querySelectorAll('a');
      var target = items[active >= 0 ? active : 0];
      if (target) { window.location.href = target.getAttribute('href'); }
    });
    document.addEventListener('click', function (e) { if (!form.contains(e.target)) { box.hidden = true; } });
  });
})();
