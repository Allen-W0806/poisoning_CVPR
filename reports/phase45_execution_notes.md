# Phase 4.5 execution notes

Only CPU reads, synthetic Gaussian derivatives, existing unit tests and new audit deliverables were executed. No new model inference, scientific optimizer update, GPU allocation or scheduler.

The first canonical audit pass stopped at an assertion that combined ID disjointness with exact-byte disjointness. Inspection found the real train/test duplicate reported as F08. The audit checker was refined to retain and enumerate content-overlap findings while keeping ID, annotation and hash checks strict; the second pass completed. This change affected only the new audit checker, not a scientific input, reward or continuation tolerance.

One read-only remote path-existence probe failed because of nested command quoting. A subsequent bounded find verified all seven requested report targets were absent before mirroring. Neither probe wrote scientific files.

Original model weights, shared runtimes, old evidence and monitor scheduler are preserved. Same-family reviewer evidence is provisional for semantic judgments.

The fresh reviewer checked all seven draft reports and requested one mathematical qualification: D affects reward within G=1, while G=0 is flat at the -1 floor. Corrected in the audit report source, all repeated finding sections and findings JSON. No scientific code changed.
