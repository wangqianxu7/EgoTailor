# Caption — fine-grained video description

Fine-grained captioning of a day's Ego4D clips with a vLLM-served Omni model.
For now the video set is **day 0**; `--day N` captions any other day.

## Files

- `start_vllm_qwen3vl.sh` — server for **Qwen3-VL-8B-Instruct** (1 card, ~0.5
  GPU mem). The lighter default for frame captioning.
- `start_vllm.sh` — server for **Qwen3-Omni-30B-A3B-Instruct** (2× A800). Heavier;
  use it when you want the bigger model.
- `caption_videos.py` — sample frames per clip, caption them in temporal
  batches, merge into one fine-grained narrative, write per-video JSON.
- `day_NN/` — output: `<video_uid>.json` per clip + `_index.json` summary.

Both servers serve on port 8080, so run only one at a time. `--model` must match
the running server's `--served-model-name`.

## Run (on the server)

```bash
# 1. start ONE model server (needs vllm + the model at MODEL_PATH)
bash Caption/start_vllm_qwen3vl.sh          # Qwen3-VL-8B  (or start_vllm.sh for Omni)
tail -f /root/vllm_server.log               # wait for "Uvicorn running on ..."

# 2. caption day 0 (server must be up; deps: analysis/requirements.txt)
pip install -r analysis/requirements.txt
python Caption/caption_videos.py --model Qwen3-VL-8B-Instruct    # match the server

# smoke test first, no server needed:
python Caption/caption_videos.py --limit 2 --dry-run
```

Output lands in `Caption/day_00/`. Re-running skips clips already captioned;
add `--overwrite` to redo them.

### Review videos against captions

To eyeball each clip next to its caption, drop the mp4 beside the json with
`--with-video` (symlink by default — instant, no disk cost):

```bash
python Caption/caption_videos.py --model Qwen3-VL-8B-Instruct \
    --out-dir /root/caption_test --with-video
```

`/root/caption_test/` then holds a matched pair per clip — `<video_uid>.mp4` and
`<video_uid>.json` — plus `_index.json`. Add `--copy` to hard-copy the mp4s
instead of symlinking (use when the folder will be moved off this machine; the
clips can be large). Start with `--limit 5` to build a small comparison set first.

## Output per clip

`day_NN/<video_uid>.json` — one whole-clip narrative plus extracted detail. The
focus fields:

- `caption` — one continuous fine-grained account of the **entire** clip.
- `camera_wearer_gender` — `male` / `female` / `unknown`, inferred from the
  wearer's **hands/forearms only** (shape, skin, nail polish, jewelry, arm hair).
  Never inferred from the activity; hands unclear or ambiguous ⇒ `unknown`.
- `preference_signals` — habit / preference hypotheses (e.g. "likes spicy
  noodles"), each with `evidence` + `confidence`; **hypotheses, not facts**.
- `actions`, `objects`, `place`, `key_details`, and the per-`segments` breakdown.

Secondary, kept only when the model actually reports them: `people_present`,
`food_and_consumables`. They are not a focus.

## Frames, not native video

The captioner samples **frames** on purpose: lifelog clips run up to 60 min, so
feeding a whole video natively would blow past the context window and lose the
"describe the whole clip" guarantee. Frames + temporal-batch stitching keep full
coverage at a predictable token cost. Audio is intentionally not used.

## Notes

- The Ego4D mp4s must exist under `--video-root`
  (default `/mnt/data_oss/raw_data/Ego4d/v2/full_scale`).
- The `--model` default (`Qwen3-Omni-30B-A3B-Instruct`) must match the server's
  `--served-model-name`. Override with `--model` or `VLLM_MODEL`.
- Fine-grained detail is bounded by frames per clip (`--frames`, default adaptive)
  and frames per call (`--batch-frames`, default 12). More frames = more detail
  and more context; keep `start_vllm.sh`'s `--max-model-len` large enough.
