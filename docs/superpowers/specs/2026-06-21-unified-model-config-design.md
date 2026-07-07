# Unified Model/Provider Configuration Design

**Date**: 2026-06-21
**Issue**: #6 — 模型切换待优化：配置繁琐缺乏统一入口

## Problem

Model/provider configuration is scattered across 4 pages:

1. `gui/settings.py` — per-stage model dropdowns (refine/vision/audio)
2. `gui/api_config.py` — provider URL, API key, default model management
3. `gui/model_manager.py` — local YOLO/CLIP model management (unchanged)
4. `gui/stage_refine.py`, `gui/stage_vision.py`, `gui/stage_audio.py` — per-tab provider dropdowns

Users must switch between 4 pages to complete model configuration. Additionally:

- **Dual API key storage**: `settings.py` saves to `user.toml [api] key`, while `api_config.py` saves to `.env`
- **Audio provider hardcoded**: `call_audio_api()` ignores the user's selection and always uses `mimo`
- **Audio stage filtering gap**: `get_available_providers("audio")` doesn't filter by `supports_audio`

## Goal

- Single settings page to configure provider + model for each stage
- Stage tabs no longer show provider/model selection
- Fix existing bugs (audio hardcoded, dual API key storage, filtering gap)

## Design

### 1. Settings Page UI Restructure

Replace the current single-provider + 3-model-combo layout in `gui/settings.py` with per-stage rows:

```
┌─ 模型配置 ──────────────────────────────────────────┐
│                                                      │
│  标题优化 (Refine)                                    │
│  Provider: [gcli          ▼]  Model: [gpt-4o   ▼] 🔄│
│                                                      │
│  视觉识别 (Vision)                                    │
│  Provider: [gcli          ▼]  Model: [gpt-4o   ▼] 🔄│
│                                                      │
│  音频识别 (Audio)                                     │
│  Provider: [mimo          ▼]  Model: [mimo-v2.5 ▼] 🔄│
│                                                      │
│  API 超时: [30] 秒                                    │
│                                                      │
│  [高级 API 配置...]                                   │
└──────────────────────────────────────────────────────┘
```

Each row: **Provider dropdown** (filtered by `get_providers_for_gui(stage)`) + **Model dropdown** (dynamically updated when provider changes) + **Fetch button**.

- Switching provider clears model and loads provider's default
- Fetch calls `/v1/models` on the selected provider
- API key input removed from Settings; users directed to "Advanced API Config" for key management

### 2. Remove Provider Selection from Stage Tabs

| File | Remove |
|------|--------|
| `gui/stage_refine.py` | `s1b_provider_var`, provider LabelFrame, dropdown (~line 55-80) |
| `gui/stage_vision.py` | `s1c_provider_var`, provider dropdown (~line 108-130) |
| `gui/stage_audio.py` | `s1ca_provider_var`, provider dropdown (~line 52-70) |

Stage tabs retain all other functionality (file selection, parameters, run buttons).

### 3. Unified Config Storage

**`config/user.toml` additions**:

```toml
[providers.stage_providers]
refine = "gcli"
vision = "gcli"
audio = "mimo"

[providers.models]
refine = "gpt-4o"
vision = "gpt-4o"
audio = "mimo-v2.5"
```

**`config/default.toml` additions**:

```toml
[providers.stage_providers]
refine = "gcli"
vision = "gcli"
audio = "mimo"
```

API keys remain in `.env` (one key per provider), managed exclusively by `api_config.py`.

### 4. Runtime Data Flow

```
Stage execution
  → load_merged_config()
  → providers.stage_providers.<stage> → provider name
  → providers.models.<stage> → model name
  → call_text_api / call_vision_api / call_audio_api(provider, model, ...)
```

Stage classes (`Refiner`, `AudioProcessor`, etc.) read provider from config instead of GUI variables.

### 5. Bug Fixes

#### 5a. Audio provider hardcoded (`providers/__init__.py:688`)

`call_audio_api()` currently hardcodes `mimo`. Fix: accept `provider_name` parameter, resolve provider config dynamically, route to correct API endpoint.

#### 5b. Audio stage filtering (`providers/__init__.py:175-178`)

`get_available_providers(stage)` only handles `"1b"` and `"1c"`. Fix: add `"audio"` case that filters by `supports_audio`.

#### 5c. Dual API key storage

Remove `api_key_var` from Settings. API keys managed exclusively via `.env` + `api_config.py`.

### Files Changed

| File | Change Type | Description |
|------|------------|-------------|
| `gui/settings.py` | Refactor | Rewrite API config section UI; per-stage provider+model rows; remove API key input |
| `gui/stage_refine.py` | Simplify | Remove provider dropdown UI and `s1b_provider_var` |
| `gui/stage_vision.py` | Simplify | Remove provider dropdown UI and `s1c_provider_var` |
| `gui/stage_audio.py` | Simplify | Remove provider dropdown UI and `s1ca_provider_var` |
| `providers/__init__.py` | Fix | `call_audio_api` accept provider param; `get_available_providers` fix audio filter |
| `core/refiner.py` | Adapt | Read provider from config, not GUI variable |
| `config/default.toml` | Update | Add `providers.stage_providers` defaults |
| `gui/api_config.py` | Minor | No longer redundant with Settings for API keys |

### Out of Scope

- `gui/model_manager.py` — local YOLO/CLIP management unchanged
- `BaseProvider` ABC refactoring — not needed for this change
- New provider additions — separate concern
