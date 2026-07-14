ESR Analysis Suite — Reference-Anchored Domain Tracking
========================================================

What is new
-----------
1. You can choose any fitted spectrum with exactly four visible domains as the
   tracking reference.
2. In the interactive browser, enter the desired domain IDs for the four fitted
   lines from left to right, then press "Set reference (T)".
3. The reference assignments are locked and the tracker propagates them both
   forward and backward in angle.
4. The browser displays persistent D1–D4 labels above the fitted components.
5. Previous and next spectra also display faint D# markers so branch motion can
   be checked visually.
6. The information panel shows whether the current spectrum is the reference,
   whether a domain is locked, and whether tracking needs review.
7. The reference and locked assignments are stored automatically in .esrsession
   files because they are part of the detailed fit-session data.

Recommended workflow
--------------------
1. Fit one sweep direction (UP or DOWN) for all angles.
2. Open the interactive fit browser.
3. Navigate to a clean spectrum where all four resonances are clearly resolved.
4. Correct/refit that spectrum first if necessary.
5. In the four boxes labelled "IDs left → right", enter a permutation of
   1, 2, 3, 4. Example: 1 2 3 4 means the leftmost line is Domain 1 and the
   rightmost is Domain 4. You may use another ordering if the physical domain
   names are already known.
6. Press "Set reference (T)" or press the T key.
7. Review the D1–D4 labels through the series. D#− indicates the previous
   spectrum and D#+ indicates the next spectrum.
8. Save the fit session after manual corrections and tracking review.

Important
---------
The program tracks resonance branches primarily by B0 continuity. Intensity is
only a weak tie-breaker because intensity is the quantity being studied. A
spectrum can have a good numerical fit but still need tracking review.
