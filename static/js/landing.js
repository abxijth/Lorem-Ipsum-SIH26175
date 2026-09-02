/* ============================================================
   DepthWizard Landing Page — reveal, nav, progress, mobile menu
   ============================================================ */

(function () {
  'use strict';

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ----------------------------------------------------------
     Scroll reveal via IntersectionObserver
     ---------------------------------------------------------- */
  if (!prefersReduced) {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );
    document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach((el) => el.classList.add('visible'));
  }

  /* ----------------------------------------------------------
     Smooth-scroll for anchor links
     ---------------------------------------------------------- */
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (!id || id === '#') return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth', block: 'start' });
      closeMenu();
    });
  });

  /* ----------------------------------------------------------
     Mobile menu toggle
     ---------------------------------------------------------- */
  const burger = document.querySelector('.nav-burger');
  const menu = document.getElementById('mobile-menu');

  function closeMenu() {
    if (!burger || !menu) return;
    burger.setAttribute('aria-expanded', 'false');
    burger.setAttribute('aria-label', 'Toggle menu');
    menu.classList.remove('open');
    menu.setAttribute('aria-hidden', 'true');
  }
  function toggleMenu() {
    if (!burger || !menu) return;
    const isOpen = burger.getAttribute('aria-expanded') === 'true';
    burger.setAttribute('aria-expanded', String(!isOpen));
    burger.setAttribute('aria-label', isOpen ? 'Toggle menu' : 'Close menu');
    menu.classList.toggle('open', !isOpen);
    menu.setAttribute('aria-hidden', String(isOpen));
  }
  if (burger && menu) {
    burger.addEventListener('click', toggleMenu);
    // close when navigating away from a menu link
    menu.querySelectorAll('a').forEach((a) => a.addEventListener('click', closeMenu));
  }

  /* ----------------------------------------------------------
     Navbar scroll state + scroll progress
     ---------------------------------------------------------- */
  const navbar = document.querySelector('.navbar');
  const progress = document.querySelector('.scroll-progress');
  if (navbar || progress) {
    let ticking = false;
    const update = () => {
      const y = window.scrollY;
      if (navbar) navbar.classList.toggle('scrolled', y > 20);
      if (progress) {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        const pct = max > 0 ? (y / max) * 100 : 0;
        progress.style.width = pct + '%';
      }
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  }
})();
