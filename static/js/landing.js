/* ============================================================
   DepthWizard — Bold Editorial Studio Edition
   Custom cursor, text reveals, marquee duplication.
   ============================================================ */

(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isCoarse = window.matchMedia('(pointer: coarse)').matches;

  /* ============================================================
     THEME — hanging light bulb. Pull down to toggle light / dark.
     ============================================================ */
  const bulbWrap = document.getElementById('bulbWrap');
  const bulbDrop = bulbWrap ? bulbWrap.querySelector('.bulb-drop') : null;
  const bulbCord = bulbWrap ? bulbWrap.querySelector('.bulb-cord') : null;
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');
  const LIGHT_THEME_COLOR = '#FFFFFF';
  const DARK_THEME_COLOR = '#0A0A0A';

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (bulbWrap) {
      var isDark = theme === 'dark';
      bulbWrap.setAttribute('aria-pressed', isDark ? 'true' : 'false');
      bulbWrap.setAttribute('aria-label', isDark
        ? 'Pull down to switch to light mode'
        : 'Pull down to switch to dark mode');
    }
    if (themeColorMeta) {
      themeColorMeta.setAttribute('content', theme === 'dark' ? DARK_THEME_COLOR : LIGHT_THEME_COLOR);
    }
  }

  function getThemePref() {
    try {
      var stored = localStorage.getItem('dw-theme');
      if (stored === 'light' || stored === 'dark') return stored;
    } catch (err) { /* ignore */ }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    var next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try {
      localStorage.setItem('dw-theme', next);
    } catch (err) { /* ignore */ }
  }

  applyTheme(getThemePref());

  if (bulbWrap) {
    var MAX_PULL = 120;
    var TOGGLE_THRESHOLD = 56;
    var startY = 0;
    var pulled = 0;
    var baseCord = 46;
    var dragging = false;

    // read the responsive base rope length from CSS so it matches the
    // current breakpoint (desktop vs mobile)
    function cordBase() {
      if (bulbCord) {
        return parseFloat(window.getComputedStyle(bulbCord).height) || 46;
      }
      return 46;
    }

    function resetBulb() {
      dragging = false;
      pulled = 0;
      bulbWrap.classList.remove('dragging');
      // clear the inline height so it animates back to the CSS --cord-base
      if (bulbCord) bulbCord.style.height = '';
    }

    bulbWrap.addEventListener('pointerdown', function (e) {
      dragging = true;
      pulled = 0;
      startY = e.clientY;
      baseCord = cordBase();
      bulbWrap.classList.add('dragging');
      bulbWrap.setPointerCapture && bulbWrap.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    bulbWrap.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      var delta = e.clientY - startY;
      if (delta < 0) delta = 0;
      pulled = Math.min(delta, MAX_PULL);
      // the rope stretches downwards; the bulb rides on its end (flex child),
      // so it stays attached to the rope as it drops.
      if (bulbCord) bulbCord.style.height = (baseCord + pulled) + 'px';
    });

    function endPull(e) {
      if (!dragging) return;
      var didToggle = pulled >= TOGGLE_THRESHOLD;
      resetBulb();
      if (didToggle) toggleTheme();
    }

    bulbWrap.addEventListener('pointerup', endPull);
    bulbWrap.addEventListener('pointercancel', resetBulb);

    // keyboard fallback
    bulbWrap.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleTheme();
      }
    });

    // Set a fallback cursor for coarse pointers via CSS; keep grab on fine pointers.
    if (isCoarse) bulbWrap.style.cursor = 'pointer';
  }

  /* ============================================================
     TEAM — single source of truth for the team roster.
     Fill in real social URLs next to the placeholders below.
     Roles are inferred; edit freely.
     ============================================================ */
  const TEAM = [
    {
      name: 'Ananuay Krishna Menon',
      role: 'ML / Pipeline',
      blurb: 'Owns the depth model pipeline and calibration — from GAMUS fine-tuning to RANSAC.',
      github: '#',   // e.g. https://github.com/username
      linkedin: '#', // e.g. https://www.linkedin.com/in/username
      email: '#',    // e.g. user@example.com
    },
    {
      name: 'Abhijith R Pillai',
      role: 'Backend / API',
      blurb: 'Built the FastAPI service, job queue, and on-demand SRTM + GeoTIFF export.',
      github: '#',
      linkedin: '#',
      email: '#',
    },
    {
      name: 'Adithya R',
      role: '3D Viewer',
      blurb: 'Shipped the Three.js flythrough and Deck.gl map view from raw height arrays.',
      github: '#',
      linkedin: '#',
      email: '#',
    },
    {
      name: 'Jeevan Manoj',
      role: 'Geospatial',
      blurb: 'Handled georeferencing, region statistics, and the DSM GeoTIFF assembly.',
      github: '#',
      linkedin: '#',
      email: '#',
    },
    {
      name: 'Kashinadh Nair',
      role: 'Validation',
      blurb: 'Ran the held-out validation and the honest "gate" experiments — what shipped, and what didn\u2019t.',
      github: '#',
      linkedin: '#',
      email: '#',
    },
    {
      name: 'Prarthana Manju Deepak',
      role: 'Frontend / UX',
      blurb: 'Crafted the interface, interactions, and this very landing experience.',
      github: '#',
      linkedin: '#',
      email: '#',
    },
  ];

  function initials(name) {
    return name
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map(function (w) { return w[0]; })
      .join('')
      .toUpperCase();
  }

  /* ============================================================
     CUSTOM CURSOR — mix-blend-mode difference
     ============================================================ */
  const cursor = document.getElementById('cursor');
  if (cursor && !isCoarse && !reduceMotion) {
    let mx = 0, my = 0;
    let cx = 0, cy = 0;

    document.addEventListener('mousemove', (e) => {
      mx = e.clientX;
      my = e.clientY;
    });

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function tick() {
      cx = lerp(cx, mx, 0.14);
      cy = lerp(cy, my, 0.14);
      cursor.style.left = cx + 'px';
      cursor.style.top = cy + 'px';
      requestAnimationFrame(tick);
    }
    tick();

    document.querySelectorAll('a, button, .project-item').forEach((el) => {
      el.addEventListener('mouseenter', () => cursor.classList.add('hovering'));
      el.addEventListener('mouseleave', () => cursor.classList.remove('hovering'));
    });
  }

  /* ============================================================
     TEXT REVEAL — slide up into view
     ============================================================ */
  const revealEls = document.querySelectorAll('.reveal-line');
  if (revealEls.length && 'IntersectionObserver' in window && !reduceMotion) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.2 }
    );
    revealEls.forEach((el) => observer.observe(el));
  } else if (reduceMotion) {
    revealEls.forEach((el) => el.classList.add('revealed'));
  }

  /* ============================================================
     MARQUEE — duplicate for seamless loop
     ============================================================ */
  const marqueeTrack = document.getElementById('marqueeTrack');
  if (marqueeTrack) {
    const cards = marqueeTrack.innerHTML;
    marqueeTrack.innerHTML += cards;
  }

  /* ============================================================
     PROJECT IMAGE GRAYSCALE REVEAL (JS fallback if needed)
     ============================================================ */
  if (reduceMotion) {
    document.querySelectorAll('.project-image img').forEach((img) => {
      img.style.filter = 'none';
      img.style.transition = 'none';
    });
  }

  /* ============================================================
     FULLSCREEN MENU TOGGLE
     ============================================================ */
  const navToggle = document.getElementById('navToggle');
  const overlayMenu = document.getElementById('overlayMenu');

  function toggleMenu(open) {
    if (!overlayMenu) return;
    if (open !== undefined) {
      overlayMenu.classList.toggle('open', open);
    } else {
      overlayMenu.classList.toggle('open');
    }
    const isOpen = overlayMenu.classList.contains('open');
    document.body.style.overflow = isOpen ? 'hidden' : '';

    if (navToggle) {
      const label = navToggle.querySelector('.nav-toggle-label');
      if (label) label.textContent = isOpen ? 'Close' : 'Menu';
      navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      navToggle.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
      const plus = navToggle.querySelector('.plus');
      if (plus) plus.style.transform = isOpen ? 'rotate(45deg)' : '';
    }
  }

  if (navToggle && overlayMenu) {
    navToggle.addEventListener('click', () => toggleMenu());

    overlayMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => toggleMenu(false));
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') toggleMenu(false);
    });
  }

  /* ============================================================
     SMOOTH ANCHOR SCROLL (for inline section links)
     Uses scrollIntoView so CSS scroll-padding-top offsets the fixed nav.
     ============================================================ */
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href');
      if (id.length < 2) return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      try {
        target.scrollIntoView({
          behavior: reduceMotion ? 'auto' : 'smooth',
          block: 'start',
        });
      } catch (err) {
        window.location.hash = id;
      }
    });
  });

  /* ============================================================
     FULL-WIDTH UNDER 560px: ensure horizontal overflow is impossible
     ============================================================ */
  if (reduceMotion) {
    document.documentElement.style.scrollBehavior = 'auto';
  }

  /* ============================================================
     TEAM — render flip cards + roster into the page
     ============================================================ */
  function buildSocial(icon, href, label) {
    if (!href || href === '#') return '';
    return '<a class="team-social" href="' + href + '" target="_blank" rel="noopener" aria-label="' + label + '">' + icon + '</a>';
  }

  const teamGrid = document.getElementById('teamGrid');
  if (teamGrid) {
    teamGrid.innerHTML = TEAM.map(function (member, i) {
      var socials = buildSocial('GH', member.github, member.name + ' on GitHub') +
        buildSocial('in', member.linkedin, member.name + ' on LinkedIn') +
        buildSocial('@', member.email, 'Email ' + member.name);
      return (
        '<button type="button" class="team-card" data-index="' + i + '" aria-pressed="false">' +
          '<span class="team-card-scene">' +
            '<span class="team-face team-front">' +
              '<span class="team-initials">' + initials(member.name) + '</span>' +
              '<span class="team-name">' + member.name + '</span>' +
              '<span class="team-role mono">' + member.role + '</span>' +
              '<span class="team-hint mono">Tap to flip</span>' +
            '</span>' +
            '<span class="team-face team-back">' +
              '<span class="team-back-role mono">' + member.role + '</span>' +
              '<span class="team-back-name">' + member.name + '</span>' +
              '<span class="team-back-blurb">' + member.blurb + '</span>' +
              '<span class="team-socials">' + (socials || '<span class="team-nosocial mono">links coming soon</span>') + '</span>' +
            '</span>' +
          '</span>' +
        '</button>'
      );
    }).join('');

    teamGrid.querySelectorAll('.team-card').forEach(function (card) {
      function flip(open) {
        var isOpen = open !== undefined ? open : !card.classList.contains('flipped');
        card.classList.toggle('flipped', isOpen);
        card.setAttribute('aria-pressed', isOpen ? 'true' : 'false');
      }
      card.addEventListener('click', function () { flip(); });
      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          flip();
        }
      });
    });

    /* Close a card when clicking anything inside it that's a social link */
    teamGrid.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('.team-social')) {
        var card = e.target.closest('.team-card');
        if (card) card.classList.remove('flipped');
      }
    });
  }

  /* Footer roster */
  var footerRoster = document.querySelector('[data-roster]');
  if (footerRoster) {
    footerRoster.textContent = TEAM.map(function (m) {
      return m.name.split(/\s+/).slice(0, 2).join(' ') + ' — ' + m.role;
    }).join('  ·  ');
  }

  /* ============================================================
     ANIMATED COUNTERS — count up on scroll
     ============================================================ */
  var statValues = document.querySelectorAll('.stat-value[data-count]');
  if (statValues.length && 'IntersectionObserver' in window && !reduceMotion) {
    function animateCount(el) {
      var target = parseFloat(el.getAttribute('data-count'));
      var decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
      var suffix = el.getAttribute('data-suffix') || '';
      var sign = el.getAttribute('data-sign') || '';
      var duration = 1400;
      var start = null;
      var unitEl = el.querySelector('.stat-unit');

      function render(value) {
        var text = sign + value.toFixed(decimals) + suffix;
        el.firstChild && el.removeChild(el.firstChild);
        if (unitEl) el.insertBefore(document.createTextNode(text), unitEl);
        else el.textContent = text;
      }
      function finalize() {
        render(target);
      }
      function frame(ts) {
        if (start === null) start = ts;
        var progress = Math.min((ts - start) / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        render(target * eased);
        if (progress < 1) requestAnimationFrame(frame);
        else finalize();
      }
      requestAnimationFrame(frame);
    }

    var counterObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCount(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    statValues.forEach(function (el) {
      if (el.getAttribute('data-suffix') === '%') el.setAttribute('data-sign', '\u2212');
      counterObserver.observe(el);
    });
  }

  /* ============================================================
     HERO BEFORE / AFTER SLIDER (clip-path version)
     ============================================================ */
  var hero = document.getElementById('heroCompare');
  if (hero) {
    var heroDrag = false;

    function heroSet(clientX) {
      var rect = hero.getBoundingClientRect();
      if (rect.width === 0) return;
      var pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * 100;
      hero.style.setProperty('--pos', pct + '%');
    }
    function heroDown(e) {
      heroDrag = true;
      e.preventDefault();
      heroSet(e.clientX !== undefined ? e.clientX : e.touches[0].clientX);
    }
    function heroMove(e) {
      if (!heroDrag) return;
      heroSet(e.clientX !== undefined ? e.clientX : e.touches[0].clientX);
    }
    function heroUp() { heroDrag = false; }

    hero.addEventListener('mousedown', heroDown);
    window.addEventListener('mousemove', heroMove);
    window.addEventListener('mouseup', heroUp);
    hero.addEventListener('touchstart', heroDown, { passive: false });
    window.addEventListener('touchmove', heroMove, { passive: false });
    window.addEventListener('touchend', heroUp);
  }

})();
