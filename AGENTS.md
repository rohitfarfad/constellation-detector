# Computer Vision Project 1: project instructions

## 1. Project purpose and scope

This workspace covers **all five questions** of NYU Tandon CS-GY 6643 Computer Vision, Project 1, Fall 2026. **Complete Q1-Q4 first**, using the methods discussed in class and the lecture slides and code supplied by the user. Then build the Q5 Kaggle constellation-detection solution on the relevant concepts and building blocks developed in that work.

**User-required organization:** Use `Q1to4Solution.ipynb` for Q1-Q4 and `ConstellationDetection.ipynb` for Q5. Keep each notebook readable and reproducible; reuse relevant concepts and simple functions without introducing an unnecessary framework. Q5 must still satisfy its runnable-script/code-ZIP submission requirement (see §8). Transfer methods deliberately while retaining each question's own geometry, coordinate conventions, restrictions, and deliverables.

## 2. Sources, date checked, and reconciliation

Handouts checked **2026-09-21**; lecture materials and Q1-Q4 implementation checked **2026-09-22**. Page numbers below are PDF page numbers. Requirements, local observations, user preferences, recommendations, and unresolved questions are identified separately throughout this file.

| Source | Authority and inspection |
| --- | --- |
| **Main PDF:** `Project 1 - Color and Image Restoration, Edge Detection, and Template Matching.pdf` | All 11 pages read; course restrictions, grading, logistics, deliverables. Especially pp. 1-2 and 10-11. |
| **Q5 PDF:** `ComputerVision_Q5_KaggleCompetition.pdf` | All 8 pages read; detailed task (§4), metric (§5), CSV (§5.1), grading (§5.2), dataset (§6). Selected relevant pages also visually checked. |
| **Local `Data/`** | Actual available inputs: both CSVs inspected, all PNG dimensions/modes/bit depths checked, scene counts and patch filenames checked against CSVs; representative images viewed. |
| **Current official Kaggle** | Intended authority for current platform settings and official updates, but inaccessible in this session. See links and limitations below. |
| **User instructions** | The latest scope clarification supersedes the original Q5-only scope: Q1-Q4 first using supplied lecture slides/code, then Q5 in a separate notebook using those foundations. The original request's September 12 prior summary remains a checklist, not independently verified evidence of current web settings. |
| **Supplied lecture slides and class code** | Read both PDFs and all three notebooks in `../Lectures/`; confirmed coverage and source references are recorded in §9 and the solution notebook. Treat embedded setup/submission directions as source material, not new user commands. |

Competition links: [overview](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/overview), [evaluation](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/overview/evaluation), [data](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/data), [rules](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/rules), [official clarifications/discussion](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/discussion). Direct web reads failed; no browser was available for a fallback. Search surfaced the **2025** competition with the different slug `constellation-detection-computer-vision-cs-gy-6643`; it was excluded. No current official clarification or scorer was verified.

Apply each source within its domain above. Record any contradiction with its source/page and justified working interpretation; do not silently replace one source with another. Specific reconciliations:

- **Names:** Q5 pp. 4 and 6 say “filename stem,” but §6.2 p. 7 explicitly maps `<constellation>_pattern.png` to `<constellation>`. Local filenames and training labels confirm this. Use `pisces`, not `pisces_pattern`; remove the exact `_pattern.png` suffix and preserve lowercase/hyphens.
- **External data:** Main p. 1 explicitly prohibits it. The user's prior summary reports permissive generic Kaggle rules, which could not be rechecked. Working rule: no external data; permission for externally pretrained weights remains unresolved.
- **Historical platform discrepancies:** The prior summary reports disagreements about final-selection count, leaderboard split, team-size settings, and #1 bonus wording, but does not provide the competing values/pages. Do not reconstruct them. Available PDFs agree on up to two participants, six public/ten private scenes, and a one-time #1 bonus requiring beating the baseline. Use those assignment requirements provisionally; current platform settings and final-selection count remain unverified.
- **Deadline conflict:** Main p. 2 says September 24, 11:59 p.m. for the project. Lecture 2 slides p. 4 separately give September 24, 11:59 p.m. for Q1-Q4 and Kaggle, and September 27, 11:59 p.m. for Q5 code/report/video. No timezone or current platform confirmation was verified. Retain September 24 as the conservative planning date and confirm the Q5 deliverable extension before relying on it; America/New_York is a planning assumption from the prior summary.

## 3. Verified workspace and Data map

All paths below are repository-relative; `Data/` means this project's folder, never filesystem `/Data`.

| Path | Contents |
| --- | --- |
| `ComputerVision_Q5_KaggleCompetition.pdf` | Q5 handout, 8 pages |
| `Project 1 - Color and Image Restoration, Edge Detection, and Template Matching.pdf` | Main handout, 11 pages |
| `Data/patterns/<name>_pattern.png` | 48 reference drawings |
| `Data/train/<name>/<name>_image.png` | Sky for `pisces`, `scorpius`, or `taurus` |
| `Data/train/<name>/patches/patch_XX.png` | Queries numbered from `01` through the supplied count |
| `Data/validation/<Id>/<Id>_image.png` | 16 competition skies |
| `Data/validation/<Id>/patches/patch_XX.png` | Corresponding queries |
| `Data/train_ground_truth.csv` | Three labeled rows |
| `Data/sample_submission.csv` | Sixteen required prediction rows |

At onboarding there was no existing AGENTS.md, code, notebook, dependency file, scorer, lecture material, or Git directory in the workspace. Other original files were `.DS_Store` metadata. Inputs comprise 851 PNGs and two CSVs. No supplied file was missing relative to the inspected CSVs.

The table above describes the **Kaggle inputs inspected during onboarding**. The separate Q1-Q4 datasets are now available:

| Path | Verified contents |
| --- | --- |
| `Q1 Data/` | `distorted_nebula.png` (1672×941 RGB), `reference_rgb_histogram.csv` (`intensity,red,green,blue`), and `reference_rgb_histogram.png` (990×505 RGB) |
| `Q2 Data/` | Five `observation_1.png` through `observation_5.png`, each 1024×810 RGBA |
| `Q3 Data/` | `degraded_spacecraft.png`, 900×900 RGBA |
| `Q4 Data/` | `search_scene.jpg`, 6000×3000; five variable-size `templates/query_*.png` |
| `Q1to4Solution.ipynb` | Executed Q1-Q4 solutions, explanations, figures, diagnostics, coordinate table, and AI prompt record |
| `outputs/` | Generated Q1/Q2 restorations, Q3 binary edges, and full-resolution annotated Q4 panorama |

Q1 inputs were supplied locally. Q2-Q4 were downloaded from the official dataset folders linked in Main pp. 5, 6, and 9; these are assignment inputs, not external supplemental data. Preserve all original inputs. Lecture files remain in `../Lectures/` and are source references, not runtime dependencies.

**Local Python environment:** `.venv/` uses Python 3.12.6 with NumPy, Matplotlib, OpenCV (`opencv-python`), Pillow, `ipykernel`, and `nbclient`; direct dependency versions are pinned in `requirements.txt`. Use `.venv/bin/python` or activate with `source .venv/bin/activate`. Install dependencies with `.venv/bin/python -m pip install -r requirements.txt`. VS Code's default interpreter is set in `.vscode/settings.json`; select the `.venv/bin/python` notebook kernel (`Python (.venv)`) if needed. The kernel is installed inside `.venv`, not globally. `os` is built in, and `google.colab.files` remains an optional Colab-only import. `.gitignore` excludes the environment and generated Python/notebook caches.

## 4. Task definition and coordinates

**Q1-Q4 requirements (Main pp. 3-9):** Solve these first in the shared Q1-Q4 notebook, with a clear section for each question.

| Question | Task and required outputs |
| --- | --- |
| Q1: Histogram matching and color recovery | Restore the distorted image using the supplied RGB histogram information; histogram matching is required. Preserve input dimensions, show input/output histograms and supporting evidence, and explain limitations. No clean reference image is supplied. |
| Q2: Multi-observation restoration | Reconstruct a 1024×810 image using meaningful information from all five observations. Establish any assumed geometric relationships from evidence; explain and support the combination method. Hidden-reference evaluation uses PSNR and SSIM; do not invent local ground-truth scores. |
| Q3: Filtering and edge detection | Produce a 900×900 single-channel edge map with only 0 and 255. Recover spacecraft boundaries/structures while suppressing degradation and star-field responses. Follow the from-scratch restriction in §8. |
| Q4: Template matching | For five queries, locate or reject an original 180×180 region in the 6000×3000 panorama. Report its **top-left** `(x, y)`, with `x` in `[0,6000)` and `y` in `[0,2820]`; absent queries get `-1` for both. Allow horizontal seam wrapping only. Include exactly five rows with columns `query_id,x,y`, plus a full-size annotated panorama with labeled boxes for accepted matches, split across the seam when needed. Justify acceptance/rejection rather than accepting every best match. |

**Requirements (Q5 pp. 2-4, §4):** For every query, report `-1` if absent; otherwise report its **center** `(x, y, m)` in the original sky image. `x` is column, `y` is row, origin `(0, 0)` is top left. `m=1` claims a figure star; `m=0` means present but off-figure. Off-figure is not absent. Predict one of the 48 supplied constellation names for each scene, or `unknown` to decline a guess; follow the name reconciliation in §2 above.

Q5 uses neither Q4's top-left region coordinates, horizontal panorama wrapping, nor its 180×180 regions (Main pp. 7-9).

**Specified difficulty (Q5 pp. 2-3):** Queries undergo arbitrary rotation, moderate scaling, subpixel shifts, blur, illumination drift, noise, and compression degradation. Geometric changes are about the patch center. All three query categories undergo the same degradation; appearance alone does not establish category. Only part of the figure is issued, with genuine off-figure distractors. Reference drawings are schematic nodes/edges with arbitrary dimensions, orientation, handedness, and aspect ratio. A strongest candidate match still needs an evidence-based acceptance/rejection decision.

## 5. Q5 dataset structure, labels, and split

**Verified local observations:** All 19 skies are 3000×3000, 8-bit grayscale PNG (`L`); all 784 queries are 32×32 with the same mode/bit depth. All 48 references are 8-bit RGBA PNGs with variable dimensions (width 143-400; height 64-400). Viewed Pisces and Ursa Major references have bright star nodes and green connecting lines; viewed skies show dense point sources, strong background variation, seams/artifacts; representative present/off-figure/absent training patches all look like small star-field crops. These are descriptive inspections, not matching results.

| Labeled scene | Queries | Figure (`m=1`) | Off-figure (`m=0`) | Absent |
| --- | ---: | ---: | ---: | ---: |
| `pisces` | 41 | 10 | 17 | 14 |
| `scorpius` | 41 | 10 | 16 | 15 |
| `taurus` | 34 | 6 | 12 | 16 |
| Total | 116 | 26 | 45 | 45 |

Validation has **16 scenes, 668 queries, 21-87 per scene**. IDs are `constellation_01` through `_08`, then `_10` through `_17`: **there is no `_09`** in either the sample CSV or local directories. Use the actual CSV row set, not a generated continuous range. Every scene's patch filenames form the expected `patch_01.png` through `patch_NN.png` sequence and match its supplied `n_patches`.

**Specified split (Q5 §6, pp. 7-8):** Three labeled scenes support local checks and are excluded from the leaderboard. `validation/` is the competition prediction set, not a labeled development set: six scenes public (approximately 40%), ten private. Which IDs belong to each is not supplied. A separate unreleased test set checks the submitted script. All target classes are among the 48 references, including classes absent from training. No validation ground truth or hidden test scenes are locally available.

## 6. Q5 evaluation and scoring implications

**Requirements (Q5 §5, pp. 4-5):**

`S = 0.25*Presence + 0.20*Localization + 0.25*GeometricRecovery + 0.30*Identification`

Average `S` equally across scenes, regardless of query count.

- **Presence:** average of present-class F1 and absent-class F1 over real query columns only.
- **Localization:** mean coordinate reward over truly present queries; predicting absent for one earns zero. For pixel distance `d`, `r(d)=1` when `d<=12`, `(36-d)/24` when `12<d<36`, and `0` when `d>=36`.
- **GeometricRecovery:** nearest-first, one-to-one matching of issued ground-truth figure stars against **all predicted present points**, using the same reward; sum rewards and divide by the number of issued figure stars. The prose states extra off-figure points cannot lower this component.
- **Identification:** 1 for the exact correct name, otherwise 0. A wrong name loses this 30% term, without erasing the other components.

**Unresolved:** The published prose does not explain how predicted `m` affects scoring, despite requiring it in the task. Do not assume it is scored separately, gates geometric matching, or is irrelevant. No executable scorer was supplied or verified. Empty-class F1 conventions, zero-denominator scenes, tie handling, and other unspecified edge cases must not be invented or presented as official behavior.

**Recommendation:** Evaluate all four components separately and report scene-level evidence. Rejecting absent queries matters independently of localization; recovering present off-figure points is legitimate. Three labeled scenes provide sanity checks, not evidence of broad generalization.

## 7. Q5 submission CSV contract

**Requirements (Q5 §5.1 pp. 5-6 and §§6.5-6.6 p. 8), verified against both local CSVs:**

- Exactly 90 columns in this order: `Id,n_patches,patch_01,...,patch_87,constellation`.
- Exactly one row for every sample-submission `Id`. Preserve its identifier, supplied `n_patches`, and preferably its row order. Map `patch_XX.png` to `patch_XX`.
- Each real query cell is `-1` or a quoted `"(x, y, m)"` tuple. Use explicit full tuples; the PDF accepts omitted `m` as zero and some alternate text formats, but those are unnecessary.
- Preserve `-1` padding beyond `n_patches`; padding is ignored by the metric, not counted as real absent queries.
- Use the exact constellation token described in §2, or `unknown`.
- All sample query cells are `-1`, and all sample names are `unknown`. This valid format-only starting point is described as scoring above zero (Q5 p. 6); its score was not measured here. **Do not equate it with the designated qualifying baseline entry.**
- Generate predictions and CSV contents programmatically; never hand-edit predictions to improve a score. Keep the original sample file immutable and write future submissions elsewhere.

## 8. Course rules, AI disclosure, grading, and deliverables

**Requirements (Main pp. 1-2, 10-11; Q5 pp. 1, 4, 6):**

- No external data. Q5 permits classical, learned, or combined methods; classical-only applies to Q1-Q4. The OpenCV prohibition is Q3-specific, not a Q5 prohibition. Pre-built libraries are otherwise allowed unless specifically restricted; externally pretrained weights require clarification.
- **Q3 specifically:** Do not use OpenCV for any part of its solution. Implement filtering, gradient computation, and edge selection from scratch with basic numerical operations based on lecture concepts. A ready-made function performing these steps from another package also violates the restriction (Main p. 1). Keep helpers reused from other questions compliant with this rule.
- No memorized scene names/coordinates, scene-specific patch-count constants, or filename/file-size shortcuts to answers. Reading filenames to locate inputs and copying supplied counts are legitimate input handling. Every coordinate and annotation must be computed programmatically.
- The runnable solution must reproduce its submission and work unchanged on unseen scenes; an unreproducible leaderboard score does not count.
- Solo or teams of two; one set of deliverables per team. Q5 p. 1 allows five Kaggle submissions per day. Earlier Kaggle submission time breaks private-leaderboard ties (Main p. 10).
- Declare AI tools and attach the sequence of user questions/prompts used (Main p. 1). Preserve actual prompts in order; never invent past prompts or substitute hidden model reasoning for the user's question chain.

**AI record:** OpenAI Codex assisted with documentation, environment setup, lecture/data inspection, and Q1-Q4 code and explanations. The notebook retains the six available user requests in order: onboarding; expanded Q1-Q4 scope; notebook setup; virtual environment; histogram comparison; and lecture-based Q1-Q4 solutions. The first prompt was read from `/Users/rohitfarfad/.codex/attachments/24c772fc-392f-4136-9e8d-3d694a2c36ce/pasted-text.txt` and is included in the notebook. Earlier prompts are unavailable and were not reconstructed. Continue the disclosure log for later assistance; no Q5 predictions or submissions have been produced.

**Project grading (Main p. 2):** Q1: 5 points; Q2: 10; Q3: 10; Q4: 10; Q5: up to 65, broken down below.

| Q5 course component | Points |
| --- | ---: |
| Report and two-minute video | 5 |
| Private leaderboard standing | 40 |
| Finishing above the designated baseline | 15 |
| Reaching #1 while beating the baseline, one-time bonus | 5 |
| Maximum | 65 |

The #1 bonus includes the final private winner and is earned at most once per team; each team member receives the credit (Main p. 10; Q5 §5.2 p. 6). The Kaggle metric is **not** directly multiplied into a 65-point course grade. Neither a rank-to-points formula nor the designated baseline identity/score is supplied in the PDFs.

**Q1-Q4 deliverable (Main pp. 1-2, 10-11):** One Google Colab notebook link submitted on Brightspace, with all code, executed outputs, final results, figures, measurements, explanations, and discussion in that notebook; no separate Q1-Q4 writeup. Include team names and NetIDs at the top. Restart and run all cells in order before submission, preserve visible outputs, and enable link sharing so graders can open it without requesting access.

**Q5 development and deliverables:** Develop Q5 in its own Python notebook as requested by the user. The notebook must support reproducible generation of the Kaggle CSV, and the eventual code ZIP must also satisfy the Q5 handout's runnable-script requirement; notebook development does not waive it. Submit the prediction CSV on Kaggle and the Q5 code ZIP on Brightspace, with a separate report following the [Overleaf template](https://www.overleaf.com/read/ddppsfrfwtss#849c73) and a separate **two-minute** approach-explanation video. Report and video stay outside the ZIP. Include team names and NetIDs in the report; no Q5 report page limit.

**Logistics:** Main p. 10 links the [team-registration spreadsheet](https://docs.google.com/spreadsheets/d/1i-zgGlv-mJR5WtygDJKVK-T7fdnUaLL-z1rAeQKuUyA/edit?usp=sharing) and [Kaggle invitation](https://www.kaggle.com/t/62875c01038f4c8eaa35cfe4041e053a) (NYU email required). Links were extracted, not acted on; the template and registration-sheet contents were not inspected. Deadline interpretation is in §2. Main p. 2 deducts one point per completed late hour; this does not establish that Kaggle accepts late submissions.

## 9. Implementation preferences and class-method coverage

**User preferences:** Keep eventual Python small, readable, sufficient, and easy to explain. Prefer simple functions and few files. Avoid unnecessary classes, frameworks, configuration systems, dependencies, and abstractions; do not compress code into cryptic expressions. Prefer a simple classical solution using class material; do not default to neural networks or pretrained models merely because Q5 permits learning.

**Confirmed class coverage (read 2026-09-22):**

| Supplied source | Relevant material |
| --- | --- |
| `CV_2026_Lecture1.pdf` | Brightness/contrast and histogram/CDF lookup tables pp. 67-86; median/padding pp. 90-96; convolution/correlation pp. 101-117; Gaussian filtering pp. 118-123; matching pp. 127-128; median pp. 130-131 |
| `Computer-Vision Lecture 2 - Fall 26.pdf` | Correlation pp. 15-17; Sobel pp. 27-29; Canny (smoothing, gradient, suppression, thresholding/linking) pp. 32-42; Harris pp. 63-105; scale/blob pp. 106-118; SIFT/matching pp. 119-133 |
| `CV_2026_Lecture 1.ipynb` | Image I/O, RGB/BGR, histograms/CDFs, brightness/contrast, box/Gaussian/bilateral filters |
| `CV_2026_Lecture 2.ipynb` | Histograms/CDFs, filters, normalized template correlation, Sobel/Canny, SIFT and ratio matching |
| `CV_2026_Lecture_3.ipynb` | Least squares, RANSAC, scale/rotation/translation/shear, similarity fitting with outlier rejection |

Use these confirmed concepts first. Q1 midpoint-CDF tie handling and Q2 robust intensity-corrected fusion are explicitly identified as adaptations of class concepts, not verbatim lecture recipes. Q3 implements Canny operations with NumPy only. Q4 combines correlation with local SIFT/RANSAC refinement. Build later Q5 work on relevant concepts while checking its different geometry and scoring requirements.

### 9.1 How the first four solutions work

The implementation source is [Q1to4Solution.ipynb](Q1to4Solution.ipynb). Locate snippets by the section heading and code identifier below; cell IDs are additional locators in the notebook JSON, not execution counts. Measured results and limitations are recorded in §12.

| Solution and code locator | Implemented steps | Output |
| --- | --- | --- |
| **Q1 — “Q1 method: match channel CDFs”**; `rgb_histogram`, `source_quantiles`, `lookup`; cell `28f4b45b` | Count 256 intensity bins per RGB channel. Normalize the reference cumulative counts. Represent each source intensity by the midpoint of its probability mass, `(cumsum(counts) - counts/2) / sum(counts)`. Use `np.searchsorted` on the reference CDF to build a monotone lookup table; apply it independently to each channel. Compare input/output histograms and mean absolute CDF differences. Equal clipped values stay tied; this does not reconstruct lost spatial or color information. | `restored_q1`, `restored_hist`; `outputs/q1_restored.png` |
| **Q2 — “Q2 Solution” / “Intensity correction and robust fusion”**; `alignment`, `corrected_views`, `weights`; cells `6c5f3ad5`, `6e41da0c` | Check Gaussian-smoothed global and regional NCC alignment. Select a tonal anchor using a median-residual noise/detail proxy, excluding heavily clipped views. Fit per-channel gain/offset on smoothed non-extreme pixels with three residual-rejection iterations; apply corrections to original pixels. Take the median of valid observations, then average using `valid / (1 + (residual / (2*spread))**2)` with a spread floor of 2. Use the anchor fallback if every view clips. Report each view's contribution and compare images/detail crops. | `restored_q2`, `contributions`; `outputs/q2_restored.png` |
| **Q3 — “Q3 Solution”**; `correlate`, `canny_numpy`; cell `75a2f916` | Implement reflect-padded correlation and a sliding-window 3×3 median in NumPy. Apply a separable Gaussian (`sigma=1.2`), Sobel gradients divided by 8, four-direction non-maximum suppression, and 8-connected hysteresis (`low=3`, `high=8`). Retain components of at least 25 pixels. Pillow handles I/O; no OpenCV or ready-made filtering/edge detector is used. | `edges_q3`, `smooth_q3`, `magnitude_q3`, `thin_q3`; `outputs/q3_edges.png` |
| **Q4 — “Q4 Solution”**; `response`, `evidence_q4`, `matches_q4`; cell `6fe8f791` | Resize queries to the specified 180×180 source region. Subtract a Gaussian background (`sigma=3`), correlate with `TM_CCOEFF_NORMED`, and search a horizontally padded panorama. Accept only NCC ≥0.30 with a ≥0.15 gap to a spatially separate peak. Refine accepted candidates using SIFT, the 0.75 descriptor-ratio test, and affine RANSAC (2-pixel threshold, at least 6 inliers and 50% inlier fraction). Transform the query center and convert to source-region top-left; constrain refinement to within 5 pixels of the coarse estimate. Generate coordinates and seam-split boxes from results. | `matches_q4`, `evidence_q4`; five-row Markdown table and `outputs/q4_annotated_panorama.png` |

### 9.2 Reusing the solutions as Q5 snippets

**Status:** The following is an implementation guide, not an implemented or validated Kaggle pipeline. Reuse small operations and their explanations in the separate Q5 notebook. Copy the needed helper or calculation with explicit arguments; do not execute the entire Q1-Q4 notebook as an import or depend on its global variables, saved answers, or output images. Preserve the completed Q1-Q4 notebook. Extract shared helpers into a small `.py` file only if actual duplication warrants it or the eventual runnable Q5 script needs them.

| Existing building block | Adaptation for Q5 | Transfer limits |
| --- | --- | --- |
| **Q1 histogram/CDF diagnostics** | Replace the RGB loop with a grayscale histogram (`np.bincount(gray.ravel(), minlength=256)`) to inspect contrast and clipping in a query and candidate crop. Use these diagnostics to decide whether photometric normalization helps matching. | Q5 supplies no clean target histogram. Do not use Q1's nebula histogram, force a small patch to match a whole-sky distribution, or infer constellation identity from intensity statistics. Histogram matching is optional and needs evidence of improved matching. |
| **Q2 alignment checks and robust intensity fitting** | Reuse the principle “establish geometric correspondence before comparing pixels.” After a candidate has been aligned, adapt the gain/offset fit and robust residual calculation to compare overlapping valid query/crop pixels. Exclude padding and clipping from that fit and record the fit residual as additional match evidence. | Q5 has one sky per scene, not five aligned observations of the same field. Do not average different queries or skies. The five-view fusion loop, anchor choice, regional grid dimensions, and zero-shift assumption do not transfer. |
| **Q3 `correlate` and Gaussian construction** | Reuse these small numerical helpers for smoothing or background estimates; Q4's float-image minus Gaussian-background expression is another direct preprocessing snippet. Evaluate whether smoothing preserves query star structure. Local-maximum selection can inform a later star-candidate detector, but that detector is new work. | Q3 deliberately removes isolated stars and short components; those stars are Q5's signal. Do not reuse its median/component pruning or binary Canny map as the default Q5 representation. Its gradient-direction suppression is not a ready-made blob-center detector. OpenCV is allowed for Q5 even though Q3 must remain NumPy-only. |
| **Q4 normalized correlation and separate-peak rejection** | Start with background-corrected grayscale matching as a diagnostic baseline. Extend candidate generation to rotation/scale hypotheses about the query center, using the class transform material. Compare the best match with genuinely different spatial candidates and retain location, score, ambiguity gap, and transform. Cache sky preprocessing and refine promising candidates rather than repeating expensive full-resolution work unnecessarily. | A fixed 32×32 template search cannot handle all specified transformations. Do not resize every Q5 patch to 180×180, wrap the sky, or copy Q4's acceptance thresholds. Merge hypotheses for the same physical location before measuring ambiguity; calibrate decisions on labeled training scenes. Mask invalid pixels introduced by rotated/scaled templates when computing a normalized comparison. |
| **Q4 SIFT/ratio matching and RANSAC refinement** | Reuse the descriptor checks and robust geometric verification when enough correspondences exist. Keep the geometric model and inlier evidence explicit; use the simplest transform supported by the candidate and class material. | A 32×32 sparse patch may provide too few reliable SIFT features. Treat insufficient correspondences as missing evidence, not proof of absence. Provide an intensity/correlation refinement path and validate it on training data. Local refinement cannot recover a location the candidate search missed. |
| **Q4 plots and output checks** | Reuse query-versus-recovered-crop panels, score diagnostics, and programmatic annotation for accepted/rejected examples. Replace Q4 assertions with Q5 bounds, center-coordinate checks, `m` values, and the exact CSV contract in §7. | Never copy Q4 coordinates, seam logic, fixed query count, or its Markdown table as the Kaggle submission format. |

**Coordinate conversion:** Q5 requires the query **center** in the original sky. For an untransformed template of width `w` and height `h`, an NCC top-left `(x0,y0)` corresponds to `(x0+(w-1)/2, y0+(h-1)/2)` under pixel-center indexing. For a fitted query-to-crop transform `T`, map the original query center through `T`, then add the crop's origin. Track resizing, rotation-canvas translation, and any pyramid scale explicitly to return to original-sky coordinates. Unlike Q4, do not subtract the source-region half-width or apply modulo wrapping. Verify the convention against training labels rather than silently rounding or introducing a half-pixel shift.

**Remaining Q5 work — constellation identity and membership:** Q1-Q4 do not solve this stage. After localizing present patches (including off-figure ones), construct and validate a geometric representation of the supplied reference drawings and recovered star positions. The class feature/transform/RANSAC concepts are starting points for comparing partial configurations with outliers. Reference drawings are schematic; do not assume their raw pixels or coordinates are directly comparable to sky crops. Establish which geometry is reliable under scale, orientation, handedness, and aspect differences before selecting a model. Derive constellation identity and `m` from geometric support, not from patch filenames, brightness, or the fact that a patch is present. A present off-figure patch remains `(x,y,0)`, not `-1`; keep the scorer ambiguity about `m` in §6 explicit.

**Suggested implementation order:**

1. In the Q5 notebook, load scenes/queries from their actual file lists and inspect the three training-label rows. Establish the grayscale representation and coordinate convention.
2. Adapt preprocessing and matching snippets first; measure presence decisions, center errors, and failure cases before adding constellation inference. Add rotation/scale handling and refinement where baseline failures justify them.
3. Choose thresholds using labeled training scenes, preferably holding one entire scene out while selecting settings on the other two. Compare each added step against the simpler baseline; three scenes provide limited evidence. Never tune to unknown validation labels or memorized answers.
4. Add the reference-geometry and membership stage, evaluate the separately documented score components, and preserve off-figure detections. Label any local scorer assumptions; do not present an approximation as the official scorer.
5. Freeze the chosen procedure, run all competition scenes programmatically, and validate the §7 CSV schema. Keep the Q5 notebook reproducible and provide the required runnable-script export when preparing deliverables. Generating a file and submitting it are separate actions.

## 10. Future workflow and validation principles

**User-required sequence:** Q1-Q4 have been implemented and locally checked in their notebook. Review those results, then develop Q5 in a separate notebook using relevant foundations. Continue applying these validation principles:

- Keep `Data/`, all `Q1 Data/` through `Q4 Data/` inputs, lecture sources, and both handouts immutable; generated outputs belong elsewhere. Inspect and reuse existing work before adding files.
- Start with the simplest justified method from confirmed class material. Add complexity only to address an observed failure; avoid speculative architecture.
- For Q1-Q4, keep the explanation, generated result, and supporting evidence together under each question. Validate the question-specific dimensions, outputs, and restrictions in §§4 and 8; run the notebook from a clean state to verify cell order and reproducibility.
- Use labeled scenes to check presence, center localization, figure recovery, and naming separately, including present off-figure versus absent cases. State the limits of three-scene evidence.
- Validate the CSV row/column contract, patch-to-column mapping, supplied counts, padding, coordinate conventions, membership values, and label vocabulary. Do not infer labels from identifiers or tune around public-leaderboard identities.
- Preserve reproducibility and concise experiment evidence: method/settings, component outcomes, failures, and rationale for significant changes. Keep decisions and verified changes concise; never claim tests, scores, or progress that did not occur.
- Before actual submission, verify current official settings and clarify material scoring ambiguities. Ensure the delivered code reproduces the CSV without manual corrections and generalizes its input handling to unseen scenes.

## 11. Unresolved questions

1. Q1-Q4 hidden clean images, edge targets, and coordinate labels are unavailable; local diagnostics do not establish hidden-reference accuracy. Team names/NetIDs still need to be added before submission. Confirm the deadline conflict in §2.
2. How does predicted `m` affect the actual scorer? What are its unspecified edge-case conventions? Obtain the official scorer or clarification when available.
3. What are the current 2026 final-submission selection limit, platform team limit, public/private split, exact closing time/timezone, and any updated host clarifications? The current web pages were inaccessible; historical disagreement details are unavailable.
4. Which leaderboard entry is the designated baseline, and how does private rank convert to the 40 course points?
5. Does the external-data prohibition also disallow externally pretrained weights? Do not assume permission; the preferred classical approach does not require them.

## 12. Current status and next authorized step

**Completed 2026-09-22:** The user's sixth request explicitly authorized solving Q1-Q4 using the supplied lectures, superseding the original onboarding-only boundary. Read all five lecture sources, obtained official missing datasets, and implemented all four solutions in `Q1to4Solution.ipynb`. Retained the earlier histogram comparison and corrected its loading cells to use consistent local paths and RGB variables. The notebook runs in order with saved figures, measurements, and computed results; output contract assertions pass.

- **Q1:** Per-channel midpoint-CDF histogram matching preserves 1672×941 dimensions. Mean absolute CDF errors decrease from approximately `(0.384, 0.195, 0.302)` to `(0.0012, 0.0190, 0.0006)`. Nearly 40% of input green values are clipped to zero; their lost ordering cannot be recovered by a lookup table.
- **Q2:** All five 1024×810 observations contribute to gain/offset-corrected robust fusion. Global shifts are zero; 33/36 regional matches have NCC ≥0.7 and displacement ≤1 pixel. Weak regional correlations are reported rather than treated as reliable geometry. Observation 3 supplies an approximate tonal anchor. Fusion preserves more detail than this blurred anchor, but its background residual is higher than the raw mean (1.33 versus 1.00); this is not a demonstrated PSNR/SSIM improvement. Some color artifacts remain.
- **Q3:** NumPy median/Gaussian filtering, Sobel, non-maximum suppression, hysteresis, and small-component rejection produce a 900×900 uint8 binary map (13,320 edge pixels). No OpenCV or ready-made filtering/edge routine is used in this question; Pillow handles only I/O.
- **Q4:** Programmatic correlation and local SIFT/affine-RANSAC yield `query_01=(2630,1280)`, `query_02=(3050,1289)`, `query_03=(5940,1430)`, with queries 04/05 rejected. These are computed predictions, not known labels; never use them as hardcoded answers. The third box crosses the horizontal seam. The notebook includes exactly five coordinate rows and a full-size annotated panorama. Resize-first correlation is not a general rotation-invariant search.

Direct dependencies are pinned, including Pillow for Q3 I/O. Input hashes are checked for immutability. No Q5 notebook, predictions, submission, registration, rule acceptance, uploads, or external messages were performed. Next work should preserve the class-based approach and separate notebook organization; Q5 remains to be developed when requested. Before course submission, add names/NetIDs, review the explanations, rerun all cells, and prepare the required Colab sharing and disclosure. This status does not claim hidden-test accuracy.

**Documentation follow-up:** The seventh available user request was: “Update the AGENTS.md to include the how the first 4 are solved and how to use these solution as a snippets to solve the kaggle compition”. Sections 9.1-9.2 now map actual notebook code to the four methods and explain proposed Q5 reuse, required adaptations, and remaining work. This request is recorded here for the AI disclosure sequence; the notebook's existing six-prompt record has not been modified in this documentation-only update.


## 13. Q5 implementation and candidate submission (2026-09-23)

This section supersedes the earlier historical statements that Q5 is unimplemented. The user authorized implementation in `ConstellationDetection.ipynb`, followed by a Kaggle CSV. The shared implementation and runnable CLI are in `constellation.py`; the notebook contains the explanation, evaluation, plots, calibration, and CSV-generation workflow. Q1-Q4 and original inputs remain unchanged.

- **Method:** Gaussian background subtraction and local contrast normalization; rotation/scale correlation plus SIFT proposals; affine ECC refinement; aligned photometric residuals; regularized five-feature logistic match calibration using released training labels. Partial reference-node matching allows reflection and affine refinement, then checks projected reference stars against the sky and nearby control locations. Geometric associations choose only computed image candidates. ECC, calibration, and whole-reference verification are adaptations beyond the literal class recipes.
- **Local evidence:** Holding one scene out of calibration gives scores 0.9718855 (Pisces), 0.9864418 (Scorpius), and 0.9519608 (Taurus), mean **0.9700960**. Shared matching/geometry settings were developed using all three scenes; this is development cross-validation, not an untouched generalization estimate. All three names are correct; candidate search retains a location within 12 pixels for 70/71 present training queries. These are handout-based local scores, not official Kaggle results, and do not establish 93–95% public/private performance.
- **Outputs:** `outputs/q5/submission_candidate.csv` has all 16 scenes and 668 real queries in the required 90-column schema. `calibration.json`, `cross_validation.json`, and `competition_diagnostics.json` retain computed evidence. Matching caches are keyed by input bytes, matching-function source, and NumPy/OpenCV versions; deleting them recomputes candidates. No predictions were hand edited.
- **Reproduction:** Run the notebook from the project directory with the `.venv` kernel, or `.venv/bin/python constellation.py --data Data --output outputs/q5/submission_candidate.csv --calibration outputs/q5/calibration.json`. A new dataset with the same patterns/validation/sample layout can use the saved public-training calibration without a training folder. Cold image matching is computationally expensive; cached reruns are faster.
- **Checks:** The complete Q5 notebook executed from a fresh kernel with saved outputs and no cell errors; it regenerated the final CSV (361 present predictions). CSV identifiers, column order, counts, padding, vocabulary, finite coordinates, bounds, and membership values were checked. Three synthetic rotation/scale/degradation checks localized centers within 0.1 pixel. SHA256 checks matched 881 original input files against predevelopment manifests. No external data or pretrained weights were used.
- **Limits:** Only three labeled scenes are available. Strong saturation can defeat localization; refinement currently rejects candidates whose required crop crosses a sky boundary. References with fewer than four extracted nodes cannot establish a reliable affine identity and are excluded from identity hypotheses. The geometry search is approximate, and membership/scorer ambiguities in §6 remain unresolved. No Kaggle upload or leaderboard measurement has occurred. Report/video/code ZIP and names/NetIDs remain separate course-submission work.

The Q5 notebook continues the AI disclosure with actual user requests 7–10: the documentation request, class-based approach request, implementation authorization, and CSV-location question. Earlier six prompts remain in the unchanged Q1-Q4 notebook.
