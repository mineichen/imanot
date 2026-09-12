# Plan `drag_tool` 

> NOTE: this file is the living spec of the drag tool — every behavior change
> to the code must update this file in the same step.

## Goal

A `DragTool` for `imanot` (`imanot/src/tool/drag.rs`) that lets the user
click- or rect-select mask pixels, then move / resize / rotate them via a
bounding-box overlay with anchors, with a live bitmap preview during gestures
and history commits on release. Two registry variants in `annotation-tool`
differ only by `pan_on_drag`: `PanDrag` (empty-space drag pans) and
`RectDrag` (empty-space drag always rect-selects).

## Core design decision: selection owns the original

A selection has a **lifetime**: it starts on click / rect-select and ends on
deselect (click on empty area, or Esc while idle). For the whole lifetime the
tool keeps the **original selected ranges**
(`snapshot: Vec<(layer_idx, SortedRanges<u32>)>`, cloned at selection time —
the "copy of the Layer when activated"). Every preview **and every commit** is
computed fresh from that original:

```
committed_or_preview = AffineTransformHeap(snapshot.spans(), &total_matrix)
```

The tool also stores one accumulated `total_matrix: Matrix3<f64>` and, per
layer, the `committed` ranges (what the last commit wrote). A new gesture only
updates `total_matrix` (`total = delta * total`); the rasterization always runs
on the pristine original, so a sequence like
*move → commit → rotate → commit → scale → commit* never chains rasterized
outputs into each other — each commit has at most **one** rasterization of
loss. Deselect drops the snapshot; the loss baked into history at that point
becomes the new baseline, and any later selection snapshots from it.

## Transform engine: `imask::AffineTransformHeap` (already pinned)

No custom rasterizer. `imask` at our pinned rev (`e756fc3`) already exports
`AffineTransformHeap::new(spans, &Matrix3<f64>)` (`nalgebra 0.35`, already in
`Cargo.lock`): per-span transformed quads, fixed-point scanline fill,
heap-merged sorted disjoint output spans, `ImageDimension` output,
`Err(PipelineError::Empty)` when fully off-screen. Upstream tests prove exact
90° rotations, gap-free arbitrary rotations (37°, 150°, 269°…), no-gap 2x
scale, area within ~15% for rotations.

`imanot` adds `nalgebra = "0.35"` as a direct dependency (forced by the public
API signature) to build matrices:

- move → translation matrix from pointer delta (image px);
- resize → scale about the opposite anchor (or bbox center), factors from
  pointer delta / bbox size; two-axis (corner) anchors preserve the aspect
  ratio with a single factor, picked by pointer slope vs. box diagonal so the
  dragged edge under the cursor stays a visible segment — unless Shift is
  held, which frees them (single-axis edge anchors always scale freely);
- rotate → rotation about bbox center from pointer-angle delta around the
  center (Shift gives the exact unsnapped angle, see `ROTATE_SNAP_DEG` below).

Step 0 of implementation is a short offline probe (`cargo doc -p imask
--offline` or a probe test: `Rect::into_spans` → identity matrix → same spans
back) proving the export exists at the pin; fallback is bumping the `imask`
git rev.

Note: `imask::cluster()` / `ClusterSpanIter` also exists (diagonal-touching
neighbor grouping, streaming). Not needed for v1 since click selects the
topmost layer, but it is the natural answer if "cluster" ever needs finer
granularity than a whole layer.

## Locked decisions

- Click = **topmost layer at pixel** (`masks.find_layer_at`, filtered by
  `AffectedLayer`).
- **Full-affine** move + resize + rotate.
- **No history writes during drag** — originals stay in history and render
  normally, live feedback is a **black transparent bitmap** like
  `BrushTool::StrokeState`.
- Commit on **drag release as Clear-committed + Add-transformed +
  Add-restored-background in one undo hook** (see Commit protocol).
- Variants differ **only by `pan_on_drag`**.
- **Esc** cancels the in-progress gesture (idle Esc clears selection).
- Debug via **trunk without SAM**.

## Selection model

- **Click**: the 8-connected cluster under the cursor (`ImaskSet::cluster` on
  the topmost layer's spans at the pixel), or deselect on empty. Clicking
  never selects the whole layer. **Shift-click** adds the cluster to the
  existing selection — across layers — instead of replacing it; clicking an
  already-covered pixel is a no-op (coverage check per layer, so double-select
  cannot duplicate entries). Before adding, the tool rebases (bakes committed
  pixels into the snapshot originals, resets the accumulated matrix, no
  history writes), so old and new content transform uniformly afterwards.
- **Rect drag**: reuse existing `RectSelection::drag_finished`; intersect the
  resulting rect with every filled layer's ranges (independent of layer), keep
  only `AffectedLayer`-accepted layers, snapshot each hit layer's in-rect
  ranges. A rect holding no pixels selects nothing (any previous selection is
  dropped) — an empty box is never shown.
- If a commit leaves nothing visible (all content moved out of the image),
  the selection is dropped as well; the clears remain a single undo step.
- Bounding box = the **contained pixels**, never the marquee: click selections
  use the cluster bounds, fresh rect selections use the union of the per-layer
  content bounds (`minbounds` over the imask-clipped spans — the dragged box
  may cover empty areas, which must not size anchors and pivots). Overlay,
  anchors, pivots and hit-testing all use the oriented selection frame (see
  note below); a selection only ever exists while it holds visible pixels, so
  an empty box is never shown.

> NOTE (superseded 2026-09-04): the AABB approach above cannot track a rotated
> shape's size — an AABB loses orientation, and re-applying the accumulated
> matrix to an already-transformed box compounds the error each gesture.
> `ImageDimension` bounds are likewise unusable for sizing (they are minimum
> or attached-ROI bounds). Instead the tool now tracks the oriented selection
> **frame** (center, half-sizes, angle) explicitly, plus the pristine original
> geometry. The rasterization matrix is rebuilt from scratch every time as
> `T(center) · R(angle) · S(size/orig_size) · T(-orig_center)`, so neither
> pixels nor geometry ever chain through lossy intermediates; resize works
> along the frame's own axes and rotation preserves size exactly. Snapshot
> bounds are tracked from the spans themselves (`spans_bbox`), never from
> attached `ImageDimension` bounds (which may cover the whole image). Rotation
> snaps the absolute angle to 1° steps (`ROTATE_SNAP_DEG`); holding Shift gives
> the exact unsnapped value.

## Interaction state machine (`handle_interaction`)

`Idle (no selection)` → click selects / empty-drag rect-selects (or pans when
`pan_on_drag && hovered layer ∉ selection`) → `Selected`. In `Selected`: hover
hit-tests inside-bbox (move), 8 anchors (resize), rotate handle (rotate) and
sets cursors; drag starts a gesture with live preview; `drag_stopped` commits
(unless matrix ~ identity, then no-op); Esc mid-gesture cancels back to last
committed state; Esc idle or click-empty deselects and drops the snapshot.
While any pointer button is down, keep `postpone_new_images = true` so `State`
can't swap images mid-gesture (existing mechanism in `state.rs`).

Gesture hit-testing (and gesture origins) must use the PRESS position
(`interact_pointer_pos - total_drag_delta`), not the current position: with
click + drag sensing, egui only fires `drag_started()` after the pointer moved
past `max_click_dist`, so the current position has typically already left
small hit targets (anchors, rotate handle) by the time the gesture begins.
Anchor hit half-size (7px) is deliberately larger than the drawn size (9px
squares).

## Preview rendering (never touches history)

Port `BrushTool::StrokeState`'s pattern: rasterize the freshly transformed
spans into a semi-transparent black `ColorImage` texture and draw it over the
image each frame the matrix changes; originals keep rendering normally from
history underneath. Overlay: dotted bbox via existing
`ImagePainter::draw_dotted_rect`, 8 square anchors + rotate circle on a stem
above top-center, ~8px screen-space hit radius via `render_scale`. Cursors
through `CursorImageSystem::set` (same base64-PNG pattern as
`RECT_CURSOR_IMAGE`): move, N-S / E-W / diagonal resizes, rotate; use
`egui::CursorIcon` built-ins wherever reachable, custom PNG only for rotate.

## Commit protocol (Clear + Add + Restore, one undo hook)

On gesture release with a changed matrix, per affected layer:

1. `new = AffineTransformHeap(original.spans(), &total_matrix)` →
   `SortedRanges` (Empty → clear-only commit);
2. push `Clear(committed_ranges)` + `Add(new_ranges)` + `Add(restored)` as
   **one hook**: first action `tracked=true`, all following `tracked=false`
   (existing `History` tracked-grouping undoes/redoes them atomically);
3. set `committed = new`, refresh bbox, store the new history revision.

The clear targets the **previously committed** ranges, not the original —
otherwise pixels an earlier gesture moved out would never be erased. If the
result is fully off-screen (`Empty`), commit clear-only and drop the selection
(nothing left to grab).

`restored` is the layer's non-selected content under the cleared footprint
(`background ∩ committed`, empty in the common no-overlap case and then
skipped). Each entry snapshots its `background` (`layer − snapshot` at
selection time, shrunk on shift-add) precisely so a commit never erases
outsiders a previous commit absorbed — by landing exactly onto them, or via
rasterization fringe: pixels outside the original selection always remain
unchanged.

## Staleness guard

Add a tiny `revision: u64` counter to `History` (bumped on `push`/`undo`/
`redo`), exposed as `MaskImage::history_revision()`. The tool records it at
selection and after each own commit; any mismatch (user hit ctrl-Z, or
anything else rewrote history mid-selection) **drops the selection**, since
snapshot/committed no longer match the mask. Known v1 limitation: switching
the active tool factory recreates the tool instance and also drops the
selection.

## Wiring & debug

`annotation-tool/src/app/tools/mod.rs::default_tools` gains
`("PanDrag", DragTool::create_factory_with(pan_on_drag=true))` and
`("RectDrag", …false)`, both `AffectedLayer::Unspecified`. Debug/iterate
without the heavy SAM dependency:

```bash
trunk build --no-default-features
# or trunk serve … ; native equivalent:
cargo run -p annotation-tool --no-default-features --features wayland
```

## Tests

- Unit (`imanot`): click selects topmost + respects `AffectedLayer`; rect
  intersects cross-layer pixels; identity gesture commits nothing;
  move/scale/rotate correctness through `AffineTransformHeap` incl.
  off-screen → `Empty`; **multi-commit-from-original**: move→commit→rotate→
  commit equals single composed-matrix transform of the original (no chaining
  loss); per-gesture hooks undo independently in order; preview writes zero
  history actions; Esc cancels without history change; external undo drops the
  selection via the revision guard.
- Regression: existing mask/history/brush suites,
  `MaskImage::add_history_action` bounds assert, workspace clippy (nursery +
  pedantic).
- Manual with the no-SAM trunk build: both variants on a multi-layer image —
  click/rect select, move, corner/edge resize, rotate, empty-drag pan-vs-rect,
  Esc cancel + Esc deselect, ctrl-Z / ctrl-shift-Z stepping gesture-by-gesture.

## Risks

- Rotation/scale rasterization is inherently lossy (area drift ~15% on
  rotations per imask's own tests); the original-anchored design bounds it to
  one resample per commit within a selection lifetime — repeated
  select→deselect cycles still accumulate, which is accepted.
- Large (fullscreen) selections make the preview texture bbox-sized;
  recompute only on matrix change and reuse the texture handle.
- `RectSelectionResult::new` rejects edge-touching rects — keep its existing
  `min(dim-1)` clamp behavior, don't expand scope.
- `GLOBAL_TOOL_MODES.md` stays untouched: `DragTool` owns its `AffectedLayer`
  like `RectTool`/`BrushTool` do today.

## Fixed bugs descriptions
- If i select two Areas on the same Layer, the resize doesnt work. It either subtracts
  the old area or sometimes unions one of the previous areas with the new one.
