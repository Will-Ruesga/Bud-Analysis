# UI Style System

This document defines one shared design language for the project.
Goal: every new feature reuses the same structures, tokens, and hierarchy.

## 1. Plan (How We Apply This)

1. Define global tokens once (`:root`): font, sizes, colors, spacing, radius, border, shadow, z-index.
2. Assign each page section a hierarchy tier (`H1` to `H6`) before styling.
3. Assign text styles only from typography tiers (`T1` to `T6`), no custom one-off text styles.
4. Reuse existing component classes for buttons, cards, panels, menus, rows, and forms.
5. If a new component is needed, add it here first, then implement it.
6. No visual changes without tier mapping and token usage.

---

## 2. Global Typography Standard (One Font Family)

Primary font family for the whole app:

- `"Trebuchet MS", "Segoe UI Variable", "Avenir Next", sans-serif`

No per-component font overrides unless explicitly documented here.

### Typography Tiers (T1-T6)

- `T1` (Display/Page title)
  - Size: `clamp(1.65rem, 2.4vw, 2.1rem)`
  - Weight: `700`
  - Italic: `no`
  - Color: `--text`
- `T2` (Section heading)
  - Size: `1.1rem`
  - Weight: `700`
  - Italic: `no`
  - Color: `--text`
- `T3` (Card heading / Subsection heading)
  - Size: `0.95rem`
  - Weight: `700`
  - Italic: `no`
  - Color: `--accent`
- `T4` (Body / Default control text)
  - Size: `0.9rem`
  - Weight: `400`
  - Italic: `no`
  - Color: `--text`
- `T5` (Secondary / Meta labels)
  - Size: `0.84rem`
  - Weight: `400`
  - Italic: `no`
  - Color: `--muted`
- `T6` (State text: warnings/help/system)
  - Size: `0.82rem`
  - Weight: `400` (or `600` for warning emphasis only)
  - Italic: `optional`
  - Color: context token (`--muted`, error color, success color)

### Text Rules

- Use bold (`700`) for structure only (titles/headings/key labels), not decoration.
- Use italic only for helper/system notes, never for primary actions.
- Action text (buttons/links in controls) uses normal weight unless explicitly primary.

---

## 3. Visual Hierarchy Model (H1-H6)

Use these 6 tiers to place surfaces and content.

- `H1` App Background
  - Page background gradients and base canvas.
  - Token: `--page-bg` — the full gradient definition. Use this (with `background-attachment: fixed`) on any surface that must visually merge with the page background (e.g. sticky headers).
- `H2` Primary Shell
  - Main page containers (`.explorer`, `.recipe-page`).
- `H3` Primary Panels
  - Main interactive blocks (filter panel, file browser, recipe article).
- `H4` Secondary Panels / Cards
  - Preview cards, grouped boxes, menus, boxed toggles.
- `H5` Interactive Controls
  - Inputs, buttons, chips, toggles, row actions.
- `H6` Ephemeral/Overlay
  - Dropdown menus, focus rings, transient overlays.

### Foreground/Background Rules

- Foreground text/icons always sit one tier above their container surface.
- Borders separate same-tier neighbors.
- Shadows are only for `H4+` when overlap is possible (menus/cards).

---

## 4. Shared Component Classes

All shared widgets are defined once in `viewer/static/css/components.css`. Page CSS files must not redefine these. To add a new widget, add it here first, then implement it in `components.css`.

### Surfaces

- `.panel` — `H3`. Strong border, white background, large radius. Use for primary panels (filter panel, file browser wrapper, bin toggle box).
- `.panel--soft` — `H4`. Lighter border, tonal background, smaller radius. Use for cards inside a panel (preview cards, grouped boxes).

### Buttons (`H5`)

One base class `.btn` with modifiers. All buttons reuse the same border, radius, font, and line-height. Modifiers only change fill, size, or color.

- `.btn` — default medium button (used for filter actions, header nav, "Add Recipe", "Open").
- `.btn--soft` — softer fill (`--control-bg-soft`). For secondary actions like "+ Add ingredient".
- `.btn--sm` — smaller padding. For row actions in dense lists.
- `.btn--lg` — larger padding + min-height. For primary form actions ("Save", "Edit recipe").
- `.btn--icon` — 2rem square. For row remove (`-`) buttons.
- `.btn--danger` — error color text. For destructive actions.
- `.btn--success` — success color text. For "Recover".

### Form controls (`H5`)

- `.input` — text input / textarea / select. One border, radius, padding, font tier (`T4`).
- `.input--lg` — larger search-style input (e.g. main recipe search).
- `.form-label` — label text (`T5`).
- `.form-helper` — helper text below a field (`T5`).
- `.form-error` — validation/error block (`T6`, error palette).

### Chips (`H5`)

- `.chip` — three-state filter chip. State is held in `data-state="neutral|include|exclude"`; CSS handles the visual.
- `.chip__icon` / `.chip__label` — sub-elements.

### Segmented control (`H5`)

- `.segment` + `.segment__btn` — pill-shaped toggle group (e.g. min/h time unit).
- Active state: add `.is-active` to the active `.segment__btn`.

### Menu / overlay (`H6`)

- `.menu` — fixed-positioned dropdown surface, JS-positioned.
- `.menu__item` — anchor or button entry inside the menu.
- `.menu__item--danger` — destructive entry.

---

## 5. Page Component Inventory + Tier Mapping

### Explorer Page

- Webpage background: `H1`
- Explorer container (`.explorer`): `H2`
- Filter panel (`.filter-panel.panel`): `H3`
- Bin toggle box (`.panel`): `H3`
- Recipe browser container (`.file-browser`): `H3`
- Recipe preview panel (`.recipe-panel`): `H4`
- Preview cards (`.preview-card.panel--soft`): `H4`
- Search bar controls (`.input--lg`, `.btn`): `H5`
- Row dropdown menu (`.menu`): `H6`

### Recipe View Page

- Webpage background: `H1`
- Recipe page container (`.recipe-page`): `H2`
- Recipe article (`.recipe-article`): `H3`
- Header nav controls (back/edit links): `H5`

### Add/Edit Recipe Page

- Webpage background: `H1`
- Recipe page container (`.recipe-page`): `H2`
- Form article panel (`.recipe-article`): `H3`
- Form group/list areas: `H4`
- Inputs/textareas/buttons: `H5`
- Dynamic row actions (`+`, `-`): `H5`

---

## 6. Scalability Rules

- Add new colors as tokens only; no hardcoded one-off color in components.
- Add new text style only by extending tiers (never local overrides first).
- Any new page must map sections to `H1-H6` before CSS implementation.
- Any new shared widget must be added to section 4 first, then implemented in `components.css`.
- Page CSS files (`explorer.css`, `recipe.css`) hold layout only — never re-style a shared component.

---

## 7. Additional Standards To Keep Cohesion

These should also be standardized going forward:

- Spacing scale (`4, 8, 12, 16, 24, 32`)
- Radius scale (`6, 8, 10, 12, 14`)
- Border thickness (`1px` default, `2px` for focus/error emphasis only)
- Z-index scale (`base`, `raised`, `menu`, `modal`)
- Motion timings (`120ms`, `180ms`, `240ms`) with ease presets
- State colors for success/warning/error/info

