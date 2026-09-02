/* ============================================================
   DepthWizard Landing Page — reveal + navigation
   ============================================================ */

(function () {
  'use strict';

  /* ----------------------------------------------------------
     Scroll reveal via IntersectionObserver
     ---------------------------------------------------------- */
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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
    document.querySelectorAll('.reveal').forEach((el) => {
      el.classList.add('visible');
    });
  }

  /* ----------------------------------------------------------
     Smooth-scroll for anchor links (fallback for older browsers)
     ---------------------------------------------------------- */
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (!id || id === '#') return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  /* ----------------------------------------------------------
     Navbar: add subtle border on scroll
     ---------------------------------------------------------- */
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    let ticking = false;
    window.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          if (window.scrollY > 20) {
            navbar.style.borderBottomColor = '#242424';
          } else {
            navbar.style.borderBottomColor = 'transparent';
          }
          ticking = false;
        });
        ticking = true;
      }
    }, { passive: true });
  }
})();
