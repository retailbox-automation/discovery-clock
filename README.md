# Discovery Clock

**Checks every claim in a written incident report against the body-worn camera recording and points to the second where it is, or is not, on the video.**

For each claim in the report it returns **On recording** (with the timecode and what is seen or said there), **Not found on recording**, or **Check this claim** (the recording shows something different or only partly supports it). Click a claim and the player jumps to that second. It never says "the officer lied": it points a public defender or prosecutor to the second of footage, and the lawyer looks.

**Demo video (2:38):** VIDEO_URL · **Built:** all code in this repository was written on Nov 4, 2026, during the Meta Global AI Developer Hackathon.

**At a glance**
- **Tested on 5 public body-camera videos** (Volusia Sheriff's Office, 39 claims): all 5 planted contradictions came back as *check this claim*, 4 of the 5 planted invented claims as *not found* (1 as *check*), and none of the 10 as *on recording*; 21 of 29 claims from the agency's own written accounts got a timecode, 3 of them flagged *check*; 8 of 8 timecodes we checked against the frames by eye matched. Answer keys were sealed by hash before the first model run ([eval/RESULTS.md](eval/RESULTS.md)).
- **Live Meta calls on screen:** `/calls` shows every request and answer with latency and cost. A live run on a 3.7-minute video makes 4 calls and costs about $0.12.
- **Four Meta models, nothing else at runtime:** Muse Spark 1.3, Muse Voice Transcribe and SAM 3.1 on the Meta Model API; Muse Spark Open (open weights) on Baseten. Plain code checks every model answer and can only move a claim to *check*.
- **Run it without a key** (recorded live answers for all five cases are in `recordings/`):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/fetch_media.py fy6dRGapw_4
.venv/bin/python -m discovery_clock replay --case fisherman-fy6dRGapw_4 --video data/media/fy6dRGapw_4.mp4
DC_REPLAY=1 .venv/bin/python -m discovery_clock serve --port 8765   # open http://127.0.0.1:8765
```

![Case page: claims next to the player](docs/screens/case-fisherman-follow.png)

## How it works

```
report text ──► Muse Spark Open (Baseten) ──► atomic claims + kind (action / speech / time / object)
video ──► Muse Voice Transcribe ──► speech turns with speakers and times
video ──► code stamps "t=N s" on every frame ──► Files API ──► Muse Spark 1.3 (video + turns + claims)
      ──► per claim: status, start_s, end_s, evidence, reason; camera clock read from the frames
      ──► code checks the model's answer (verify.py) ──► result.json ──► web page
SAM 3.1 on a short window ──► follow a person by click · find faces to blur in an exported clip
```

1. **Claims** (`claims.py`). Muse Spark Open splits the report into atomic claims. In evaluation mode each line is one claim and the model only labels the kind. If Baseten is unavailable, Muse Spark 1.3 does the same job; the page shows which model did it.
2. **Speech** (`meta.py`). Muse Voice Transcribe in diarization mode. Long audio is cut into pieces of at most 9:30 and the times are shifted back to the video clock.
3. **Placing claims** (`review.py`). One Muse Spark 1.3 call gets the video, the transcribed turns and the claims and returns JSON (strict schema). The same call reads the date and time the body camera burns into the frame (Axon overlay) at the start, at the end and at every camera switch.
4. **Time stamps** (`stamp.py`). In the first live test the model placed a claim at 1124 s in a 222 s video: video models read text in a frame well but lose track of elapsed time. So before upload the code burns `t=N s` into the top-left corner of every frame and the prompt tells the model to read times from it. After that change all timecodes on the test video fell inside the recording, and the ones we checked against the frames by eye matched. The player shows the original video; only the uploaded copy is stamped.
5. **Code checks the model** (`verify.py`). The model's answer is never shown as is:
   - a timecode outside the recording → *check*;
   - a quoted line of speech must exist in the transcript (fuzzy match ≥ 0.8 by words). Near the timecode: fine. Elsewhere in the recording: the status stays and the claim's window is widened to that line. Nowhere: the quote is invented → *check*;
   - *on recording* without visual or speech evidence, or with evidence that says the thing is not shown → *check*;
   - a clock time in the claim (for example "around 8:55 p.m.") is compared with the camera clock at the claim's timecode; more than 10 minutes apart → *check*, with both times in the note;
   - a claim the model did not answer → *check*.

   The code can only move a claim towards *check*, never to *on recording*. Every change is written next to the claim, and the model's original answer is kept. On the page a code change appears as an amber note above the evidence.
6. **SAM 3.1** (`track.py`). Only a window around the moment is sent, at 10 frames per second, because SAM video is billed per frame.
   - *Follow a person*: pause, click a person, and SAM tracks them through the window (point prompt, both directions). The box follows them during playback.
   - *Export clip with bystander faces blurred*: SAM finds the faces in the claim's window (concept prompt `face`); the code blurs each face frame by frame, except the person being followed, and keeps the audio.

## Models and where they run

| Model | Where | Used for |
|---|---|---|
| Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API | placing claims on the video, reading the camera clock |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API | speech with speakers and times |
| SAM 3.1 (`sam-3.1`) | Meta Model API | following a person, finding faces to blur |
| Muse Spark Open (`meta-models/muse-spark-open-1026`) | Baseten (open weights) | splitting the report into claims |

**No non-Meta model is called at runtime.** Everything else is plain code: verification, time stamps, blurring (Pillow), video cutting (ffmpeg).

## Live Meta calls on screen

`/calls` shows every request and answer as it happens: time (UTC), model, endpoint, the request in short, the answer (up to 2,000 characters), latency and cost. It refreshes every 2 seconds. Keys are never logged. Rows marked **REPLAY** re-serve a recorded live answer.

![Live Meta calls](docs/screens/calls.png)

## Run it

Python 3.12, ffmpeg on PATH.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/fetch_media.py            # the public videos from the eval-media release (sha256 checked), YouTube as fallback
```

**Without a key (judges):** every live answer used for the five evaluation cases is recorded in `recordings/`. Replay runs the full pipeline, including the code checks, with no network:

```bash
.venv/bin/python -m discovery_clock replay --case fisherman-fy6dRGapw_4 --video data/media/fy6dRGapw_4.mp4
DC_REPLAY=1 .venv/bin/python -m discovery_clock serve --port 8765      # http://127.0.0.1:8765
```

Replaying all five evaluation cases this way gives exactly the numbers in `eval/RESULTS.md` (`python eval/run.py --results runs`).

A replay checks that each request matches the recorded one (hash of the full request, with uploaded files identified by the sha256 of the video). If a request differs — changed code, or a download of the video that is not byte-identical — the recorded answer for that step is still used and `/calls` marks the row "request changed since recording". In replay, SAM 3.1 answers exist for the recorded actions only: on the fisherman case, *Follow a person* at 00:02:40 and at 00:03:02 (the deputy on the dock; the second one is the live take shown in the video) and *Export clip* for claim 2.

**With a key:**

```bash
export MODEL_API_KEY=...            # Meta Model API
export BASETEN_API_KEY=...          # optional; without it Muse Spark 1.3 splits the claims
export DC_CAP_USD=3                 # spend cap; the next live call stops once the ledger reaches it
.venv/bin/python -m discovery_clock run --case my-case --video path/to/video.mp4 --report report.txt
.venv/bin/python -m discovery_clock run --case fisherman-fy6dRGapw_4 --video data/media/fy6dRGapw_4.mp4 \
    --report eval/cases/fisherman-fy6dRGapw_4/claims.txt --claims lines
.venv/bin/python -m discovery_clock serve --port 8765     # upload a video and a report in the browser
```

Results go to `runs/<case>/result.json`; the call log to `runs/<case>/calls.jsonl`; the spend ledger to `runs/ledger.jsonl`. `python -m discovery_clock reverify --case <id>` re-runs the code checks over the saved model answer without calling any model.

Tests: `.venv/bin/python -m pytest` (71 tests, no network).

## Evaluation

Five published body-camera videos of the Volusia Sheriff's Office (see *Data and rights*). For each video:

- **account** claims: sentences from the official written account the agency published with the video (names replaced);
- **planted** claims, written by us before any model run: one the recording contradicts (`planted_contradiction`) and one that is not in it (`planted_invented`). A planted claim must never come back as *on recording*.

The keys (`eval/cases/<id>/key.json`) carry these labels only, no expected timecodes. They were sealed before the first model run: `eval/SEALED.txt` holds `sha256(json.dumps(key, sort_keys=True))` for each case, and `eval/run.py` re-checks every seal and writes `eval/RESULTS.md`.

One live run per case, 39 claims (from `eval/RESULTS.md`):

| | result |
|---|---|
| planted claims kept off "on recording" | **10 / 10** |
| … planted contradictions flagged "check this claim" | **5 / 5** |
| … planted invented claims marked "not found" | 4 / 5 (1 marked "check") |
| official-account claims that got a timecode | 21 / 29 |
| … of which marked "check this claim" (listed in RESULTS.md with evidence and timecode for a human look) | 3 |
| official-account claims not found in the published clip | 8 |

**Rule change after run 1, disclosed.** In the first run the quote check moved a claim to *check* whenever a quoted line was more than 10 s from the claim's timecode. Housefire C3 quoted a real line spoken 18 seconds earlier and was wrongly moved to *check*. Rule v2: a quote found anywhere in the transcript keeps the status and widens the claim's window to that line; only a quote found nowhere moves the claim to *check*. All five cases were re-checked with v2 over the saved model answers (`python -m discovery_clock reverify`), without new model calls. The only change: housefire C3, *check* → *on recording* at 00:00:53–00:01:19. `eval/RESULTS.md` shows the as-run and v2 numbers side by side.

Spend for the five live runs: $0.07–$0.13 per case for videos of 1.6 to 4.3 minutes (Muse Spark 1.3 plus Voice Transcribe; Spark Open on Baseten credits). SAM 3.1 costs $0.02–$0.024 per 10–12 second window at 10 fps.

## Data and rights

All videos come from the official YouTube channel of the Volusia Sheriff's Office (@VolusiaSheriff). They are public records under Florida law (Florida Statutes ch. 119), released by the agency; Florida agencies cannot claim copyright in public records (*Microdecisions, Inc. v. Skinner*, 889 So. 2d 871, Fla. 2d DCA 2004).

| case | video |
|---|---|
| fisherman-fy6dRGapw_4 | [Fisherman falls out of wheelchair into water - rescued!](https://www.youtube.com/watch?v=fy6dRGapw_4) |
| housefire-phHtxd9Ln3Q | [House fire rescue: 5/18/26](https://www.youtube.com/watch?v=phHtxd9Ln3Q) |
| i4driver-iqxWGjr1PRk | [I-4 reckless drunk driver](https://www.youtube.com/watch?v=iqxWGjr1PRk) |
| dog-kCu9d3LMN6A | [Dog saved from starving, owner arrested](https://www.youtube.com/watch?v=kCu9d3LMN6A) |
| motorcycle-r-ub7DZN8BA | [Fleeing motorcyclist arrest](https://www.youtube.com/watch?v=r-ub7DZN8BA) |

- No media is stored in git. The five evaluation videos are attached, byte-identical to the files the recorded runs were made on, to the release [`eval-media`](https://github.com/retailbox-automation/discovery-clock/releases/tag/eval-media); `scripts/fetch_media.py` downloads them from there, checks the sha256 listed in `data/sources.json`, and falls back to YouTube (yt-dlp) only if that fails.
- Names of arrested people are replaced in the claims.
- Bystanders' faces are blurred in anything we export or publish (SAM 3.1 finds the faces, the code blurs them).

## Limitations

- Five videos, 39 claims, one run each. That is a sanity check, not a benchmark.
- We wrote the evaluation keys ourselves (the labels and the planted claims); they were sealed by hash before the first run, but they are ours.
- The statuses are pointers for a human reviewer, not findings. *Not found* means the model did not find the claim in this clip; another camera or the full footage may show it.
- Muse Voice Transcribe returns turns, not word-level timestamps, so a quote is located to the turn (a few seconds), not to the word.
- Voice Transcribe takes up to 10 minutes per file; longer audio is cut into 9:30 pieces. One Muse Spark call per video has been tested here only on clips of up to about 4.3 minutes; much longer recordings will need splitting.
- Times in claims are compared with the camera clock only when the claim states a clock time.
- SAM tracking works on a short window around the clicked moment (4 s before, 8 s after).

## Layout

```
discovery_clock/   meta.py (client, call log, spend cap, record/replay) · claims.py · stamp.py · review.py
                   verify.py · track.py (SAM) · pipeline.py · cli.py · app.py + templates/ + static/
eval/              cases/<id>/{claims.txt,key.json} · SEALED.txt · run.py · RESULTS.md
recordings/<case>/ recorded live answers (replay without a key)
scripts/           fetch_media.py · run_eval_cases.sh
tests/             pytest, no network
```

License: MIT.
