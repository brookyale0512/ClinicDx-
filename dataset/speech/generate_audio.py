#!/usr/bin/env python3
"""
Voice Scribe — Kokoro TTS Audio Generation
==========================================
Generates audio training data for the MedASR→MedGemma projector.

Architecture:
  - 16 parallel worker processes, each with its own Kokoro instance
  - Each clip gets 2-5 randomly selected voices from the 28-voice pool
  - Clip size distribution matches natural doctor dictation
  - All 54,971 CIEL concepts guaranteed to appear in at least one clip
  - Negative examples (NOT_IN_MANIFEST, NONE) included at 10%

Clip size distribution:
  1 phrase:  50%  — concept isolation training
  2-3:       20%  — short dictation
  4-5:       15%  — typical encounter
  6-7:        8%  — full encounter
  8:          2%  — maximum
  NEGATIVE:   5%  — NOT_IN_MANIFEST + NONE examples

Output:
  /medASR/data/audio/
    clips/          — one WAV per (clip_id, voice)
    clips.jsonl     — clip definitions with manifest + expected output
    manifest.jsonl  — maps clip_id → {voice: wav_path}
    checkpoint.json — resume state

Usage:
    python3 generate_audio.py --test 10      # test with 10 clips
    python3 generate_audio.py --clips 2000000 # full run
    python3 generate_audio.py --resume        # resume after interruption
    python3 generate_audio.py --progress      # show dashboard only
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Constants ─────────────────────────────────────────────────────────────────

AUDIO_DIR     = Path("/medASR/data/audio")
TRAINING_FILE = Path("/medASR/data/scribe_training/scribe_training.jsonl")
SAMPLE_RATE   = 24000  # Kokoro native output; train_projector.py resamples to 16kHz on load
SILENCE_MS    = 400
WORKERS       = 16

VOICE_POOL = [
    "af_alloy","af_aoede","af_bella","af_heart","af_jessica","af_kore",
    "af_nicole","af_nova","af_river","af_sarah","af_sky",
    "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael",
    "am_onyx","am_puck","am_santa",
    "bf_alice","bf_emma","bf_isabella","bf_lily",
    "bm_daniel","bm_fable","bm_george","bm_lewis",
]

# Clip size distribution
CLIP_DIST = {
    1: 0.50,
    2: 0.10,
    3: 0.10,
    4: 0.08,
    5: 0.07,
    6: 0.05,
    7: 0.03,
    8: 0.02,
    "neg": 0.05,  # negative examples
}

MANIFEST_SIZE = (25, 45)  # min/max manifest lines per clip

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name, log_file=None):
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    for n in ("httpx","httpcore","hpack"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logger

# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(path: Path):
    """Load all training pairs, group by concept for efficient sampling."""
    logger = logging.getLogger("main")
    logger.info("Loading training data from %s ...", path)
    t0 = time.time()

    pairs = []
    by_ciel = defaultdict(list)
    by_class = defaultdict(list)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                pairs.append(r)
                by_ciel[r.get("ciel_id","")].append(r)
                by_class[r.get("concept_class","?")].append(r)
            except: pass

    logger.info("Loaded %d pairs | %d concepts | %d classes | %.1fs",
                len(pairs), len(by_ciel), len(by_class), time.time()-t0)
    return pairs, by_ciel, by_class

# ── Clip assembly ─────────────────────────────────────────────────────────────

def build_manifest(target_ciel_ids: list, by_ciel: dict,
                   all_manifest_lines: list,  # pre-built flat list
                   rng: random.Random) -> list:
    """Build manifest: target concepts + random fillers, shuffled."""
    target_lines = set()
    for cid in target_ciel_ids:
        pairs = by_ciel.get(cid, [])
        if pairs:
            ml = pairs[0].get("manifest_line","")
            if ml: target_lines.add(ml)

    filler_count = rng.randint(*MANIFEST_SIZE) - len(target_lines)
    filler_count = max(0, filler_count)

    fillers = []
    if filler_count > 0 and all_manifest_lines:
        # Sample from pre-built list — O(n) not O(876K)
        sampled = rng.sample(all_manifest_lines, k=min(filler_count, len(all_manifest_lines)))
        fillers = [ml for ml in sampled if ml not in target_lines]

    manifest = list(target_lines) + fillers
    rng.shuffle(manifest)
    return manifest


def assemble_clip(clip_idx: int, size: int, all_pairs: list,
                  by_ciel: dict, by_class: dict,
                  all_manifest_lines: list,
                  rng: random.Random, n_voices: int) -> dict:
    """Build one clip definition."""
    class_list = list(by_class.keys())

    # Select pairs from different classes when possible
    selected = []
    cls_order = class_list.copy()
    rng.shuffle(cls_order)
    for cls in cls_order:
        if len(selected) >= size: break
        pool = by_class[cls]
        if pool: selected.append(rng.choice(pool))
    while len(selected) < size:
        selected.append(rng.choice(all_pairs))
    rng.shuffle(selected)

    phrases = [r["phrase"] for r in selected]
    target_ids = [r.get("ciel_id","") for r in selected]
    output_lines = [r.get("expected_output","") for r in selected]
    manifest = build_manifest(target_ids, by_ciel, all_manifest_lines, rng)

    voices = rng.sample(VOICE_POOL, k=min(n_voices, len(VOICE_POOL)))
    clip_id = f"clip_{clip_idx:08d}"

    return {
        "clip_id":      clip_id,
        "phrase_count": size,
        "phrases":      phrases,
        "voices":       voices,
        "manifest":     "CONCEPTS:\n" + "\n".join(manifest),
        "output":       "\n".join(output_lines),
        "negative":     False,
        "concepts": [{
            "ciel_id":       r.get("ciel_id",""),
            "concept_class": r.get("concept_class",""),
            "manifest_line": r.get("manifest_line",""),
            "fhir_type":     r.get("fhir_type",""),
        } for r in selected],
    }


def assemble_negative_clip(clip_idx: int, all_pairs: list, by_ciel: dict,
                            all_manifest_lines: list,
                            rng: random.Random, n_voices: int) -> dict:
    """Build a negative example clip."""
    neg_type = rng.choice(["not_in_manifest", "none"])
    voices   = rng.sample(VOICE_POOL, k=min(n_voices, len(VOICE_POOL)))
    clip_id  = f"clip_{clip_idx:08d}"

    if neg_type == "none":
        # Ambient phrase with no clinical content
        none_phrases = [
            "The patient is here today",
            "Good morning",
            "Please come in",
            "How are you feeling",
            "Take a seat please",
            "Let me check your file",
            "When did this start",
            "Thank you for coming",
        ]
        phrase  = rng.choice(none_phrases)
        manifest = build_manifest([], by_ciel, all_manifest_lines, rng)
        return {
            "clip_id":      clip_id,
            "phrase_count": 1,
            "phrases":      [phrase],
            "voices":       voices,
            "manifest":     "CONCEPTS:\n" + "\n".join(manifest),
            "output":       "NONE",
            "negative":     True,
            "neg_type":     "none",
        }
    else:
        # Phrase mentions concept NOT in manifest
        pair = rng.choice(all_pairs)
        phrase = pair["phrase"]
        excluded_id = pair.get("ciel_id","")
        concept_name = pair.get("manifest_line","").split("]")[-1].split("(")[0].strip()

        # Build manifest excluding this concept
        excl_line = pair.get("manifest_line","")
        excl_lines = [ml for ml in all_manifest_lines if ml != excl_line]
        count = rng.randint(*MANIFEST_SIZE)
        manifest_lines = rng.sample(excl_lines, k=min(count, len(excl_lines)))
        rng.shuffle(manifest_lines)

        return {
            "clip_id":      clip_id,
            "phrase_count": 1,
            "phrases":      [phrase],
            "voices":       voices,
            "manifest":     "CONCEPTS:\n" + "\n".join(manifest_lines),
            "output":       f"NOT_IN_MANIFEST: {concept_name}",
            "negative":     True,
            "neg_type":     "not_in_manifest",
        }


def generate_all_clips(n_clips: int, all_pairs: list, by_ciel: dict,
                       by_class: dict, clips_path: Path,
                       seed: int = 42, test_mode: bool = False) -> int:
    """Stream-write clips to file as they're built. Returns total clip count."""
    # Pre-build flat deduplicated manifest line list once — O(n) sampling
    all_manifest_lines = list({p.get("manifest_line","") for p in all_pairs
                                if p.get("manifest_line","")})
    print(f"  Manifest line pool: {len(all_manifest_lines):,} unique lines")
    """Generate all clip definitions with guaranteed concept coverage.

    test_mode=True: skip coverage pass, just build n_clips random clips fast.
    """
    rng = random.Random(seed)
    total_written = 0

    sizes_no_neg   = [k for k in CLIP_DIST if k != "neg"]
    weights_no_neg = [CLIP_DIST[k] / (1 - CLIP_DIST["neg"]) for k in sizes_no_neg]
    neg_quota = max(1, int(n_clips * CLIP_DIST["neg"]))

    t0 = time.time()
    with open(clips_path, "w") as f:

        if not test_mode:
            # Pass 1: guarantee every concept appears at least once — stream write
            all_ciel_ids = list(by_ciel.keys())
            rng.shuffle(all_ciel_ids)
            covered = set()
            coverage_count = 0

            for ciel_id in all_ciel_ids:
                if ciel_id in covered: continue
                pairs = by_ciel[ciel_id]
                if not pairs: continue
                size     = rng.choices([1,2,3,4], weights=[0.5,0.2,0.2,0.1], k=1)[0]
                n_voices = rng.randint(2, 3)
                clip = assemble_clip(total_written, size, all_pairs,
                                     by_ciel, by_class, all_manifest_lines, rng, n_voices)
                f.write(json.dumps(clip, ensure_ascii=False) + "\n")
                covered.update(c["ciel_id"] for c in clip["concepts"])
                total_written += 1
                coverage_count += 1

            print(f"  Coverage pass: {coverage_count:,} clips cover {len(covered):,} concepts "
                  f"({time.time()-t0:.0f}s)")

        # Pass 2: fill remaining quota — stream write
        remaining  = n_clips - total_written
        pos_count  = remaining - neg_quota

        for i in range(pos_count):
            size     = rng.choices(sizes_no_neg, weights=weights_no_neg, k=1)[0]
            n_voices = rng.randint(2, 3)
            clip = assemble_clip(total_written, size, all_pairs, by_ciel, by_class,
                                 all_manifest_lines, rng, n_voices)
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")
            total_written += 1
            if total_written % 200_000 == 0:
                rate = total_written / max(time.time()-t0, 1)
                eta  = (n_clips - total_written) / max(rate, 1)
                print(f"  Assembled {total_written:,}/{n_clips:,} clips "
                      f"({rate:.0f}/sec, ETA {eta:.0f}s)")
                f.flush()

        for i in range(neg_quota):
            n_voices = rng.randint(2, 3)
            clip = assemble_negative_clip(total_written, all_pairs, by_ciel,
                                          all_manifest_lines, rng, n_voices)
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")
            total_written += 1

    elapsed = time.time() - t0
    print(f"  Assembly complete: {total_written:,} clips in {elapsed:.0f}s "
          f"({total_written/elapsed:.0f} clips/sec)")
    return total_written

# ── Audio synthesis (runs in worker process) ──────────────────────────────────

def worker_process(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                   out_dir: Path, log_dir: Path):
    """Worker process: loads Kokoro, processes clips from queue."""
    import warnings
    warnings.filterwarnings("ignore")

    log_file = log_dir / f"worker_{worker_id:02d}.log"
    logger   = setup_logger(f"worker_{worker_id:02d}", log_file)
    logger.info("Worker %02d starting...", worker_id)

    try:
        import torch
        from kokoro import KPipeline
        pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        gpu_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info("Worker %02d ready | GPU: %.0f MB | device: %s",
                    worker_id, gpu_mb, next(pipe.model.parameters()).device)
    except Exception as e:
        logger.error("Worker %02d failed to load Kokoro: %s", worker_id, e)
        result_queue.put({"worker": worker_id, "error": str(e)})
        return

    silence = np.zeros(int(SAMPLE_RATE * SILENCE_MS / 1000), dtype=np.float32)
    generated = failed = 0

    while True:
        try:
            task = task_queue.get(timeout=30)
        except Exception:
            break

        if task is None:  # poison pill
            break

        clip_id = task["clip_id"]
        phrases = task["phrases"]
        voice   = task["voice"]
        out_path = out_dir / f"{clip_id}_{voice}.wav"

        if out_path.exists():
            result_queue.put({"worker": worker_id, "clip_id": clip_id,
                              "voice": voice, "path": str(out_path),
                              "skipped": True})
            continue

        try:
            # Synthesize each phrase
            parts = []
            ok = True
            for i, phrase in enumerate(phrases):
                chunks = []
                for _, _, audio in pipe(phrase, voice=voice, speed=1.0):
                    if audio is not None and len(audio) > 0:
                        chunks.append(audio)
                if not chunks:
                    ok = False; break
                parts.append(np.concatenate(chunks).astype(np.float32))
                if i < len(phrases) - 1:
                    parts.append(silence)

            if not ok or not parts:
                failed += 1
                result_queue.put({"worker": worker_id, "clip_id": clip_id,
                                  "voice": voice, "failed": True})
                continue

            audio = np.concatenate(parts)
            sf.write(str(out_path), audio, SAMPLE_RATE, subtype='PCM_16')
            duration = len(audio) / SAMPLE_RATE
            size_kb  = out_path.stat().st_size / 1024
            generated += 1

            result_queue.put({
                "worker":   worker_id,
                "clip_id":  clip_id,
                "voice":    voice,
                "path":     str(out_path),
                "duration": round(duration, 2),
                "size_kb":  round(size_kb, 1),
                "phrases":  len(phrases),
            })

            if generated % 100 == 0:
                logger.info("Worker %02d | gen=%d fail=%d", worker_id, generated, failed)

        except Exception as e:
            failed += 1
            logger.warning("Worker %02d | clip=%s voice=%s error: %s",
                           worker_id, clip_id, voice, e)
            result_queue.put({"worker": worker_id, "clip_id": clip_id,
                              "voice": voice, "failed": True})

    logger.info("Worker %02d DONE | generated=%d failed=%d", worker_id, generated, failed)
    result_queue.put({"worker": worker_id, "done": True,
                      "generated": generated, "failed": failed})


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"done_keys": [], "generated": 0, "failed": 0, "started": None}
    try:
        d = json.loads(path.read_text())
        d["done_keys"] = set(d.get("done_keys", []))
        return d
    except:
        return {"done_keys": set(), "generated": 0, "failed": 0, "started": None}

def save_checkpoint(ckpt: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    out = dict(ckpt)
    out["done_keys"] = list(ckpt["done_keys"])
    out["last_updated"] = datetime.now().isoformat()
    tmp.write_text(json.dumps(out, indent=2))
    tmp.rename(path)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def print_dashboard(stats: dict, clear: bool = True, stream=None):
    if stream is None:
        stream = sys.stderr

    lines = []
    now      = datetime.now().strftime("%H:%M:%S")
    elapsed  = time.time() - stats.get("t0", time.time())
    gen      = stats.get("generated", 0)          # new files THIS run
    failed   = stats.get("failed", 0)
    prev     = stats.get("prev_done", 0)          # files from previous run(s)
    pending  = stats.get("pending", 0)             # work to do THIS run
    total    = stats.get("total_tasks", 0)
    done_this_run = gen + failed
    pct      = done_this_run / max(pending, 1) * 100
    rate     = gen / max(elapsed, 1)
    remaining = max(0, pending - done_this_run)
    eta_s    = remaining / max(rate, 0.001) if rate > 0.1 else 0
    disk_gb  = stats.get("disk_gb", 0)
    total_on_disk = prev + gen
    w        = 40
    filled   = min(w, int(pct / 100 * w))
    bar      = "█" * filled + "░" * (w - filled)

    lines.append(f"{'═'*70}")
    lines.append(f"  Voice Scribe TTS  [{now}]  workers={stats.get('active_workers',0)}")
    lines.append(f"{'═'*70}")
    lines.append(f"  [{bar}] {pct:.1f}%")
    lines.append(f"  This run:  {gen:>8,} / {pending:,}  (failed: {failed})")
    lines.append(f"  On disk:   {total_on_disk:>8,} / {total:,} total")
    lines.append(f"  Rate:      {rate:>8.1f} files/sec = {rate*3600:,.0f}/hr")
    lines.append(f"  Elapsed:   {str(timedelta(seconds=int(elapsed))):>12}")
    lines.append(f"  ETA:       {str(timedelta(seconds=int(eta_s))):>12}")
    lines.append(f"  Disk used: {disk_gb:.1f} GB")
    lines.append(f"{'─'*70}")

    worker_stats = stats.get("workers", {})
    if worker_stats:
        lines.append(f"  Workers:")
        for wid in sorted(worker_stats.keys()):
            ws = worker_stats[wid]
            status = "✓ done" if ws.get("done") else "⚡ running"
            lines.append(f"    [{wid:02d}] gen={ws.get('gen',0):>5,} fail={ws.get('fail',0):>3,} {status}")

    samples = stats.get("samples", [])
    if samples:
        lines.append(f"{'─'*70}")
        lines.append(f"  Recent outputs:")
        for s in samples[-3:]:
            lines.append(f"    {s['clip_id']} | {s['voice']:15} | {s.get('phrases',1)}ph | "
                         f"{s.get('duration',0):.1f}s | {s.get('size_kb',0):.0f}KB")

    lines.append(f"{'═'*70}")

    # Move cursor to top-left and overwrite in-place (stderr = terminal)
    if clear:
        stream.write("\033[2J\033[H")
    stream.write("\n".join(lines) + "\n")
    stream.flush()


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_generation(clips: list, out_dir: Path, log_dir: Path,
                   n_workers: int, resume: bool, ckpt_path: Path,
                   logger: logging.Logger):
    """Orchestrate parallel generation across N workers."""

    ckpt = load_checkpoint(ckpt_path) if resume else \
           {"done_keys": set(), "generated": 0, "failed": 0, "started": datetime.now().isoformat()}

    done_keys = set(ckpt["done_keys"]) if isinstance(ckpt["done_keys"], (list, set)) else set()

    # Build task list: one task per (clip, voice)
    all_tasks = []
    for clip in clips:
        for voice in clip["voices"]:
            key = f"{clip['clip_id']}_{voice}"
            all_tasks.append({
                "clip_id": clip["clip_id"],
                "phrases": clip["phrases"],
                "voice":   voice,
                "key":     key,
            })

    pending = [t for t in all_tasks if t["key"] not in done_keys]
    logger.info("Total tasks: %d | Already done: %d | Pending: %d",
                len(all_tasks), len(done_keys), len(pending))

    if not pending:
        logger.info("All tasks already complete!")
        return

    # Queues — task queue large enough to hold all pending tasks
    task_q   = mp.Queue(maxsize=len(pending) + n_workers + 10)
    result_q = mp.Queue()

    # Start workers
    workers = []
    for wid in range(n_workers):
        p = mp.Process(target=worker_process,
                       args=(wid, task_q, result_q, out_dir, log_dir),
                       daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.3)  # stagger startup

    logger.info("Started %d worker processes", n_workers)

    # Pre-load ALL tasks into queue before starting collection loop
    logger.info("Loading %d tasks into queue...", len(pending))
    for task in pending:
        task_q.put(task)
    # Send poison pills
    for _ in range(n_workers):
        task_q.put(None)
    logger.info("All %d tasks + %d poison pills queued", len(pending), n_workers)

    # Stats — separate "previous run" from "this run" counters
    prev_done    = len(done_keys)               # files from previous run(s)
    prev_gen     = ckpt.get("generated", 0) if resume else 0
    stats = {
        "t0":            time.time(),
        "total_tasks":   len(all_tasks),        # all (clip, voice) pairs
        "prev_done":     prev_done,             # already on disk before this run
        "generated":     0,                     # new files THIS run
        "failed":        0,                     # failures THIS run
        "pending":       len(pending),           # work to do THIS run
        "active_workers": n_workers,
        "workers":       {i: {"gen": 0, "fail": 0} for i in range(n_workers)},
        "samples":       [],
        "disk_gb":       0.0,
    }

    # Manifest file for this run
    manifest_path = AUDIO_DIR / "manifest.jsonl"

    # Feed tasks + collect results
    results_done    = 0
    workers_done    = 0
    last_ckpt_save  = time.time()
    last_dash       = time.time()
    last_disk_check = time.time()

    _shutdown = False
    def handle_signal(sig, frame):
        nonlocal _shutdown
        _shutdown = True
        logger.info("Shutdown requested — draining queue...")
    signal.signal(signal.SIGINT, handle_signal)

    with open(manifest_path, "a") as mf:
        while workers_done < n_workers:

            # Collect results
            while True:
                try:
                    result = result_q.get(timeout=0.05)
                except:
                    break

                wid = result.get("worker", 0)

                if result.get("done"):
                    workers_done += 1
                    stats["active_workers"] = n_workers - workers_done
                    stats["workers"][wid]["done"] = True
                    logger.info("Worker %02d finished | gen=%d fail=%d",
                                wid, result.get("generated",0), result.get("failed",0))
                    continue

                if result.get("error"):
                    logger.error("Worker %02d error: %s", wid, result["error"])
                    workers_done += 1
                    continue

                key = f"{result.get('clip_id','')}_{result.get('voice','')}"
                done_keys.add(key)
                results_done += 1

                if result.get("failed"):
                    stats["failed"] += 1
                    stats["workers"][wid]["fail"] = stats["workers"][wid].get("fail",0) + 1
                elif result.get("skipped"):
                    stats["generated"] += 1  # already existed on disk, counts as done
                else:
                    stats["generated"] += 1
                    stats["workers"][wid]["gen"] = stats["workers"][wid].get("gen",0) + 1
                    stats["samples"].append(result)
                    if len(stats["samples"]) > 10:
                        stats["samples"] = stats["samples"][-10:]
                    # Write to manifest
                    mf.write(json.dumps({
                        "clip_id":  result["clip_id"],
                        "voice":    result["voice"],
                        "path":     result["path"],
                        "duration": result.get("duration", 0),
                        "size_kb":  result.get("size_kb", 0),
                        "phrases":  result.get("phrases", 1),
                    }, ensure_ascii=False) + "\n")
                    mf.flush()

            # Disk usage check every 60s (use du, not rglob — fast on large dirs)
            if time.time() - last_disk_check > 60:
                try:
                    import subprocess
                    result_du = subprocess.run(
                        ["du", "-sb", str(out_dir)],
                        capture_output=True, text=True, timeout=10)
                    if result_du.returncode == 0:
                        stats["disk_gb"] = int(result_du.stdout.split()[0]) / 1e9
                except: pass
                last_disk_check = time.time()

            # Dashboard every 5s (writes to stderr, not log file)
            if time.time() - last_dash > 5:
                print_dashboard(stats)
                last_dash = time.time()

            # Checkpoint every 60s
            if time.time() - last_ckpt_save > 60:
                ckpt["done_keys"]  = list(done_keys)
                ckpt["generated"]  = prev_gen + stats["generated"]
                ckpt["failed"]     = stats["failed"]
                save_checkpoint(ckpt, ckpt_path)
                last_ckpt_save = time.time()

    # Final checkpoint
    ckpt["done_keys"] = list(done_keys)
    ckpt["generated"] = prev_gen + stats["generated"]
    ckpt["failed"]    = stats["failed"]
    save_checkpoint(ckpt, ckpt_path)

    # Final dashboard
    print_dashboard(stats, clear=False)
    logger.info("Generation complete | generated=%d (total on disk=%d) failed=%d",
                stats["generated"], prev_gen + stats["generated"], stats["failed"])

    # Wait for workers
    for p in workers:
        p.join(timeout=10)


# ── Progress-only mode ────────────────────────────────────────────────────────

def show_progress_only():
    ckpt_path = AUDIO_DIR / "checkpoint.json"
    clips_path = AUDIO_DIR / "clips.jsonl"

    if not ckpt_path.exists():
        print("No checkpoint found. Has generation started?")
        return

    ckpt = json.loads(ckpt_path.read_text())
    done = len(ckpt.get("done_keys", []))
    gen  = ckpt.get("generated", 0)
    fail = ckpt.get("failed", 0)

    total_tasks = 0
    if clips_path.exists():
        with open(clips_path) as f:
            for line in f:
                if line.strip():
                    try:
                        clip = json.loads(line)
                        total_tasks += len(clip.get("voices", []))
                    except: pass

    disk_gb = 0
    wav_dir = AUDIO_DIR / "clips"
    if wav_dir.exists():
        disk_gb = sum(f.stat().st_size for f in wav_dir.rglob("*.wav") if f.is_file()) / 1e9

    pct = done / max(total_tasks, 1) * 100
    w = 40; filled = int(pct/100*w)
    bar = "█"*filled + "░"*(w-filled)

    print(f"\n{'═'*65}")
    print(f"  Voice Scribe TTS Progress")
    print(f"{'═'*65}")
    print(f"  [{bar}] {pct:.1f}%")
    print(f"  Generated: {gen:,} | Failed: {fail:,} | Total: {done:,}/{total_tasks:,}")
    print(f"  Disk used: {disk_gb:.1f} GB")
    print(f"  Last updated: {ckpt.get('last_updated','?')}")
    print(f"{'═'*65}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voice Scribe TTS Generator")
    parser.add_argument("--clips",    type=int, default=2_000_000,
                        help="Total clips to generate (default: 2M)")
    parser.add_argument("--test",     type=int, default=0,
                        help="Test mode: generate N clips only")
    parser.add_argument("--workers",  type=int, default=WORKERS,
                        help=f"Parallel workers (default: {WORKERS})")
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress dashboard only")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    if args.progress:
        show_progress_only(); return

    n_clips  = args.test if args.test else args.clips
    log_dir  = AUDIO_DIR / "logs"
    out_dir  = AUDIO_DIR / "clips"
    ckpt_path = AUDIO_DIR / "checkpoint.json"

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    logger = setup_logger("main", AUDIO_DIR / "logs" / "main.log")
    logger.info("=" * 65)
    logger.info("Voice Scribe TTS Generator")
    logger.info("Clips: %d | Workers: %d | Voices: %d | Seed: %d",
                n_clips, args.workers, len(VOICE_POOL), args.seed)
    logger.info("Output: %s", AUDIO_DIR)
    logger.info("=" * 65)

    # Load data
    all_pairs, by_ciel, by_class = load_training_data(TRAINING_FILE)

    # Assemble clips
    clips_path = AUDIO_DIR / "clips.jsonl"
    if clips_path.exists() and args.resume:
        logger.info("Using existing clips from %s", clips_path)
    else:
        logger.info("Assembling %d clips (streaming to disk)%s...", n_clips,
                    " [TEST MODE]" if args.test else "")
        generate_all_clips(n_clips, all_pairs, by_ciel, by_class, clips_path,
                           args.seed, test_mode=bool(args.test))
        logger.info("Assembly complete → %s", clips_path)

    # Load clips (needed for stats + task building)
    logger.info("Loading clip definitions...")
    clips = []
    with open(clips_path) as f:
        for line in f:
            if line.strip():
                try: clips.append(json.loads(line))
                except: pass
    logger.info("Loaded %d clips", len(clips))

    # Stats
    total_files = sum(len(c["voices"]) for c in clips)
    size_dist   = Counter(c["phrase_count"] for c in clips)
    neg_count   = sum(1 for c in clips if c.get("negative"))
    avg_voices  = total_files / max(len(clips), 1)

    logger.info("\nCLIP SUMMARY:")
    logger.info("  Total clips:   %d", len(clips))
    logger.info("  Total files:   %d (avg %.1f voices/clip)", total_files, avg_voices)
    logger.info("  Negative clips: %d (%.1f%%)", neg_count, neg_count/len(clips)*100)
    logger.info("  Size distribution:")
    for size in sorted(size_dist.keys()):
        pct = size_dist[size]/len(clips)*100
        logger.info("    %d phrase(s): %d clips (%.1f%%)", size, size_dist[size], pct)

    # Disk estimate
    avg_kb = 130 * 0.5 + 450 * 0.5  # mix of single and multi
    est_gb = total_files * avg_kb * avg_voices / 1024 / 1024
    logger.info("  Est. disk: ~%.0f GB", est_gb)

    if args.test:
        logger.info("\nTEST MODE: %d clips", n_clips)
        # Show first 3 clips in detail
        for i, clip in enumerate(clips[:3]):
            logger.info("\n--- Clip %d ---", i+1)
            logger.info("  clip_id:  %s", clip["clip_id"])
            logger.info("  phrases:  %s", clip["phrases"])
            logger.info("  voices:   %s", clip["voices"])
            logger.info("  output:   %s", clip["output"])
            logger.info("  negative: %s", clip.get("negative", False))
            logger.info("  manifest: %s...", clip["manifest"][:100])

    # Run generation
    run_generation(clips, out_dir, log_dir, args.workers, args.resume,
                   ckpt_path, logger)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
