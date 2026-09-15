# Shared frame window and run-directory naming for the object-pose-feedback
# experiment.  SOURCED, not executed:
#
#   source "$ROOT/streaming_couping/object_pose_feedback_env.zsh"
#
# The caller must have set ROOT and STORAGE_ROOT first.  An interpreter is
# resolved here if the caller has none: the inner pipeline script names its
# interpreters SAM_PYTHON and HORIZON_PYTHON and has no PYTHON at all, so
# assuming one aborts the run under ``set -u`` before it reaches the GPU.
#
# Every command file used to spell the window out in its own run-directory
# path, so changing the window meant editing eight files and getting one of
# them wrong was silent -- a reader would simply find no runs and report
# nothing.  The window lives here once, and the directory name is derived from
# it, so the two cannot disagree.

#: How many frames the experiment uses, and where they start.  Overriding the
#: count picks a different window, and with it a different set of run
#: directories -- which is how an earlier window is read back:
#:
#:   OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh commands_check_...
#:
#: The readers default to the window named here, so pointing one at last
#: round's results is a variable, not an edit.
FRAME_COUNT="${OBJECT_POSE_FEEDBACK_FRAME_COUNT:-150}"
FRAME_START="${OBJECT_POSE_FEEDBACK_FRAME_START:-90}"
FRAME_STRIDE="${OBJECT_POSE_FEEDBACK_FRAME_STRIDE:-1}"
FRAME_END=$(( FRAME_START + (FRAME_COUNT - 1) * FRAME_STRIDE ))

#: Used for the cheap JSON checks below, and only for those.  Left empty when
#: there is none -- deciding that here would abort the source mid-file and
#: leave the functions below undefined, which reports "command not found"
#: instead of "no interpreter".
PYTHON="${PYTHON:-${STREAMING_COUPING_PYTHON:-$(command -v python || true)}}"

RUNS_ROOT="$STORAGE_ROOT/outputs"
#: Which generation the readers default to.  The newest that has runs, unless
#: a caller names one; see opf_generations below.
GENERATIONS=(v1 v2 v3 v4 v5)

#: The name every run of this window shares.  A different window is a different
#: prefix, so runs of two windows never collide and a reader can never pick up
#: a cache or a summary belonging to the other one.
RUN_PREFIX="semantic_map_${FRAME_COUNT}frames_horizonstream_object_pose_feedback_${FRAME_START}_${FRAME_END}"

opf_run_dir() {
  # opf_run_dir <generation> [branch]
  local generation="$1"
  local branch="${2:-}"
  if [[ -n "$branch" ]]; then
    print -r -- "$RUNS_ROOT/${RUN_PREFIX}_${generation}.${branch}"
  else
    print -r -- "$RUNS_ROOT/${RUN_PREFIX}_${generation}"
  fi
}

opf_generations_with_runs() {
  # Generations, oldest first, that have at least a baseline run directory.
  local generation
  for generation in "${GENERATIONS[@]}"; do
    [[ -d "$(opf_run_dir "$generation" baseline)" ]] && print -r -- "$generation"
  done
}

opf_newest_generation() {
  local found
  found=("${(@f)$(opf_generations_with_runs)}")
  (( ${#found} > 0 )) && print -r -- "${found[-1]}"
}

opf_require_frame_window() {
  # opf_require_frame_window <manifest> <scene-id>
  #
  # The frame selection SILENTLY TRUNCATES: asking for 150 frames from a scene
  # that has 120 gives 120, and the run directory would still say 150.  Every
  # number after that would be labelled with a window it did not use, so this
  # refuses instead.  CPU only, and it runs before anything touches the GPU.
  local manifest="$1" scene_id="$2" available
  if [[ -z "$PYTHON" ]]; then
    print -u2 "cannot check the frame window: no python interpreter found"
    print -u2 "  set STREAMING_COUPING_PYTHON to one"
    return 1
  fi
  available=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c "
import json, sys
with open(sys.argv[1]) as handle:
    manifest = json.load(handle)
for scene in manifest.get('scenes', ()):
    if str(scene.get('scene_id')) == sys.argv[2]:
        print(len(scene.get('frames', ())))
        break
else:
    print(-1)
" "$manifest" "$scene_id" 2>/dev/null)
  if [[ -z "$available" || "$available" == "-1" ]]; then
    print -u2 "cannot determine the frame count of scene $scene_id in $manifest"
    return 1
  fi
  if (( available < FRAME_START + FRAME_COUNT )); then
    print -u2 "scene $scene_id has ${available} frames; the requested window"
    print -u2 "  start=$FRAME_START count=$FRAME_COUNT needs at least $(( FRAME_START + FRAME_COUNT ))"
    print -u2 "the selection truncates silently, so this run would be labelled"
    print -u2 "  with a window it did not use.  Lower the count or move the start."
    return 1
  fi
  print "frame window: ${FRAME_COUNT} frames, ${FRAME_START}..${FRAME_END} (scene has ${available})"
  return 0
}
